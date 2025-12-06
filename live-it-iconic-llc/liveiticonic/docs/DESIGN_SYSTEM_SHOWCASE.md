# Live It Iconic - Design System Showcase

**Version:** 1.0.0
**Last Updated:** 2025-11-12
**Purpose:** Comprehensive reference for all design elements, components, and patterns

---

## Table of Contents

1. [Brand Identity](#brand-identity)
2. [Color System](#color-system)
3. [Typography](#typography)
4. [Logo Collection](#logo-collection)
5. [Brandmarks](#brandmarks)
6. [UI Components](#ui-components)
7. [Spacing & Layout](#spacing--layout)
8. [Motion & Animation](#motion--animation)
9. [Elevation & Shadows](#elevation--shadows)
10. [Design Patterns](#design-patterns)
11. [Template Guidelines](#template-guidelines)
12. [Accessibility Standards](#accessibility-standards)

---

## Brand Identity

### Brand Essence
**Live It Iconic** is a premium automotive lifestyle brand embodying:
- **Luxury**: Championship-grade materials and finishes
- **Performance**: Speed, precision, and excellence
- **Lifestyle**: Aspirational living and iconic moments
- **Exclusivity**: Limited editions and premium experiences

### Brand Personality
- **Sophisticated** yet approachable
- **Bold** yet refined
- **Modern** yet timeless
- **Athletic** yet luxurious

---

## Color System

### Primary Brand Colors

#### Championship Gold (#d4af37)
```css
/* Primary accent color */
color: #d4af37;
background: #d4af37;

/* Tailwind */
bg-lii-gold
text-lii-gold
border-lii-gold
```

**Usage:**
- Primary CTAs and buttons
- Premium accents and highlights
- Product badges and tags
- Interactive elements (hover states)

**Variations:**
- **Gold Press** (#c19d2e): Active/pressed states
- **Gold Hover** (#e0c050): Hover states

#### Carbon Black (#0B0B0C)
```css
/* Primary background */
background: #0B0B0C;

/* Tailwind */
bg-lii-bg
```

**Usage:**
- Primary background
- Dark mode base
- Premium containers

### Neutral Palette

| Color | Hex | Tailwind Class | Usage |
|-------|-----|----------------|-------|
| **Ink** | #14161A | `bg-lii-ink` | Secondary backgrounds, cards |
| **Charcoal** | #36454f | `bg-lii-charcoal` | Component backgrounds |
| **Graphite** | #3A4048 | `bg-lii-graphite` | Subtle backgrounds |
| **Ash** | #8C93A3 | `text-lii-ash` | Body text, secondary content |
| **Cloud** | #E6E9EF | `text-lii-cloud` | Headings, primary content |
| **Snow** | #F8F9FA | `text-white` | Pure white accents |

### Semantic Colors

| Purpose | Color | Hex | Usage |
|---------|-------|-----|-------|
| **Success** | Green | #10B981 | Confirmations, positive states |
| **Warning** | Amber | #F59E0B | Warnings, alerts |
| **Error** | Signal Red | #E03A2F | Errors, critical states |
| **Info** | Blue | #3B82F6 | Information, neutral notices |

### Color Usage Guidelines

**DO:**
- ✅ Use Gold for premium CTAs and highlights
- ✅ Maintain 4.5:1 contrast ratio for text
- ✅ Use neutral palette for backgrounds and structure
- ✅ Reserve Signal Red for errors only

**DON'T:**
- ❌ Use Gold for backgrounds (too bright)
- ❌ Mix warm and cool grays
- ❌ Use pure white (#ffffff) except for Snow
- ❌ Use more than 3 colors per component

### Color Combinations

```css
/* Luxury Card */
background: #14161A; /* Ink */
border: 1px solid rgba(212, 175, 55, 0.1); /* Gold 10% */
color: #E6E9EF; /* Cloud */

/* Premium Button */
background: #d4af37; /* Gold */
color: #0B0B0C; /* Carbon Black */

/* Subtle Text */
background: #0B0B0C; /* Carbon Black */
color: #8C93A3; /* Ash */
```

---

## Typography

### Font Families

#### Display Font: Playfair Display
```css
font-family: 'Playfair Display', serif;
```

**Usage:**
- Hero headings
- Display text
- Luxury product names
- Premium messaging

**Example:**
```html
<h1 class="font-display text-6xl font-bold">
  Live It Iconic
</h1>
```

#### UI Font: Inter Variable
```css
font-family: 'Inter Variable', sans-serif;
```

**Usage:**
- Body text
- UI elements
- Navigation
- Product descriptions

**Example:**
```html
<p class="font-ui text-base text-lii-ash">
  Premium automotive lifestyle merchandise
</p>
```

#### Monospace: JetBrains Mono
```css
font-family: 'JetBrains Mono', monospace;
```

**Usage:**
- Code snippets
- Technical specifications
- Order numbers
- API documentation

### Type Scale

| Size | Rem | Pixels | Tailwind | Usage |
|------|-----|--------|----------|-------|
| **9xl** | 8rem | 128px | `text-9xl` | Hero displays |
| **8xl** | 6rem | 96px | `text-8xl` | Massive displays |
| **7xl** | 4.5rem | 72px | `text-7xl` | Large displays |
| **6xl** | 3.75rem | 60px | `text-6xl` | Display headings |
| **5xl** | 3rem | 48px | `text-5xl` | Hero headings |
| **4xl** | 2.25rem | 36px | `text-4xl` | Page headings |
| **3xl** | 1.875rem | 30px | `text-3xl` | Section headings |
| **2xl** | 1.5rem | 24px | `text-2xl` | Component headings |
| **xl** | 1.25rem | 20px | `text-xl` | Small headings |
| **lg** | 1.125rem | 18px | `text-lg` | Large body |
| **base** | 1rem | 16px | `text-base` | Body text |
| **sm** | 0.875rem | 14px | `text-sm` | Small text |
| **xs** | 0.75rem | 12px | `text-xs` | Captions |

### Typography Patterns

#### Hero Heading
```html
<h1 class="font-display text-6xl md:text-8xl font-bold tracking-tight text-lii-cloud">
  Championship Performance
</h1>
```

#### Section Heading
```html
<h2 class="font-display text-3xl md:text-4xl font-semibold text-lii-cloud">
  Featured Collection
</h2>
```

#### Body Text
```html
<p class="font-ui text-base leading-relaxed text-lii-ash">
  Premium automotive lifestyle merchandise for those who live iconic.
</p>
```

#### Caption
```html
<span class="font-ui text-sm text-lii-ash uppercase tracking-wider">
  Limited Edition
</span>
```

### Font Weights

| Weight | Value | Tailwind | Usage |
|--------|-------|----------|-------|
| **Black** | 900 | `font-black` | Ultra-bold displays |
| **Extrabold** | 800 | `font-extrabold` | Strong emphasis |
| **Bold** | 700 | `font-bold` | Headings, CTAs |
| **Semibold** | 600 | `font-semibold` | Subheadings |
| **Medium** | 500 | `font-medium` | UI elements |
| **Normal** | 400 | `font-normal` | Body text |
| **Light** | 300 | `font-light` | Subtle text |

### Letter Spacing

| Name | Value | Tailwind | Usage |
|------|-------|----------|-------|
| **Widest** | 0.1em | `tracking-widest` | All-caps labels |
| **Wider** | 0.05em | `tracking-wider` | Uppercase headings |
| **Wide** | 0.025em | `tracking-wide` | Loose headings |
| **Normal** | 0 | `tracking-normal` | Body text |
| **Tight** | -0.025em | `tracking-tight` | Display headings |
| **Tighter** | -0.05em | `tracking-tighter` | Large displays |

---

## Logo Collection

### Premium Logos (Main Brand)

#### 1. IconicLogo
**Location:** `src/components/IconicLogo.tsx`

**Usage:**
```tsx
import { IconicLogo } from '@/components/IconicLogo';

<IconicLogo
  variant="primary" // or "mono", "white"
  size="lg" // or "sm", "md", "xl"
/>
```

**Variants:**
- **Primary**: Full color with gold accents
- **Mono**: Single color (white/black)
- **White**: All white for dark backgrounds

#### 2. ElegantLogo
**Location:** `src/components/ElegantLogo.tsx`

**Usage:**
```tsx
import { ElegantLogo } from '@/components/ElegantLogo';

<ElegantLogo size={64} color="#d4af37" />
```

**Style:** Refined, luxury serif treatment

#### 3. PremiumLogo
**Location:** `src/components/PremiumLogo.tsx`

**Usage:**
```tsx
import { PremiumLogo } from '@/components/PremiumLogo';

<PremiumLogo variant="compact" size="md" />
```

**Style:** Contemporary, sophisticated

### Icon Logos (16 variants)

All located in `src/components/icons/`:

| Logo | File | Style | Best For |
|------|------|-------|----------|
| **Apex** | ApexLogo.tsx | Sharp, angular | Performance products |
| **Aurora** | AuroraLogo.tsx | Flowing, organic | Premium collections |
| **Diamond** | DiamondLogo.tsx | Geometric, luxury | Exclusive items |
| **Elevation** | ElevationLogo.tsx | Upward, aspirational | Premium tier |
| **Infinity** | InfinityLogo.tsx | Continuous, timeless | Legacy products |
| **Nexus** | NexusLogo.tsx | Connected, modern | Tech products |
| **Nova** | NovaLogo.tsx | Radiant, explosive | New launches |
| **Phoenix** | PhoenixLogo.tsx | Bold, mythical | Special editions |
| **Prestige** | PrestigeLogo.tsx | Elegant, refined | Luxury line |
| **Pulse** | PulseLogo.tsx | Dynamic, energetic | Athletic wear |
| **Radiant** | RadiantLogo.tsx | Bright, glowing | Premium highlights |
| **Summit** | SummitLogo.tsx | Peak, achievement | Championship gear |
| **Velocity** | VelocityLogo.tsx | Fast, streamlined | Racing collection |
| **Vortex** | VortexLogo.tsx | Circular, powerful | Dynamic products |
| **Zenith** | ZenithLogo.tsx | Highest point | Top-tier items |

**Usage Example:**
```tsx
import { ApexLogo } from '@/components/icons/ApexLogo';

<ApexLogo size={48} color="#d4af37" className="hover:scale-110" />
```

### Bird Logos (Seasonal Collection)

**Location:** `src/components/logo/`

#### 1. Elegant Flamingo
- **Season**: January-March
- **Characteristics**: Grace, Refinement, Iconic Silhouette
- **Usage**: Spring collection, elegant products

#### 2. Majestic Pelican
- **Season**: April-June
- **Characteristics**: Strength, Endurance, Bold Presence
- **Usage**: Summer collection, durable products

#### 3. Caribbean Frigate Bird
- **Season**: July-September
- **Characteristics**: Freedom, Power, Dynamic Energy
- **Usage**: Fall collection, performance products

#### 4. Tropical Tanager
- **Season**: October-December
- **Characteristics**: Vibrancy, Exclusivity, Tropical Spirit
- **Usage**: Winter collection, vibrant products

**Usage Example:**
```tsx
import { BirdLogoShowcase } from '@/components/BirdLogoShowcase';

<BirdLogoShowcase variant="carousel" />
```

### Automotive Crest Logo

**Location:** `src/components/logo/AutomotiveCrestLogo.tsx`

**Usage:**
```tsx
import { AutomotiveCrestLogo } from '@/components/logo/AutomotiveCrestLogo';

<AutomotiveCrestLogo size={80} variant="gold" />
```

**Variants:**
- **Gold**: Championship gold
- **Silver**: Platinum silver
- **Mono**: Single color

---

## Brandmarks

### Typography-Based Brandmarks

All located in `src/components/brandmarks/`:

| Brandmark | Style | Best For |
|-----------|-------|----------|
| **ArtDecoStyle** | Vintage luxury, geometric | Heritage products |
| **ArchitecturalBlock** | Modern, structured | Contemporary collections |
| **BoldStatement** | Strong, impactful | Hero sections |
| **ClassicEmblem** | Traditional, refined | Established products |
| **DiagonalDynamic** | Energetic, modern | Performance gear |
| **FuturisticType** | Cutting-edge, sleek | Tech products |
| **GeometricInitials** | Clean, minimalist | Modern collections |
| **HandwrittenSignature** | Personal, authentic | Limited editions |
| **InitialsCircle** | Contained, elegant | Product badges |
| **LuxurySerif** | Classic, sophisticated | Premium products |
| **MinimalistStack** | Simple, modern | Clean designs |
| **ModernWordmark** | Contemporary, bold | Brand presence |
| **MonogramShield** | Protected, prestigious | Exclusive items |
| **ScriptElegance** | Flowing, refined | Elegant collections |
| **TechMonospace** | Technical, precise | Specifications |

**Usage Example:**
```tsx
import { LuxurySerif } from '@/components/brandmarks/LuxurySerif';

<LuxurySerif
  text="LII"
  color="#d4af37"
  size="lg"
/>
```

### Brandmark Selection Guide

**For Premium Products:**
- LuxurySerif
- ClassicEmblem
- MonogramShield

**For Modern Collections:**
- ModernWordmark
- GeometricInitials
- MinimalistStack

**For Performance Products:**
- DiagonalDynamic
- BoldStatement
- FuturisticType

**For Limited Editions:**
- HandwrittenSignature
- ArtDecoStyle
- InitialsCircle

---

## UI Components

### Radix UI Component Library

All components located in `src/components/ui/`:

#### Navigation & Menus
- **Accordion** - Collapsible content sections
- **Breadcrumb** - Navigation trail
- **Context Menu** - Right-click menus
- **Dropdown Menu** - Dropdown navigation
- **Menubar** - Application menu
- **Navigation Menu** - Main navigation
- **Sidebar** - Collapsible sidebar

#### Overlays & Dialogs
- **Alert Dialog** - Modal confirmation dialogs
- **Dialog** - Modal windows
- **Drawer** - Slide-out panels
- **Popover** - Floating content containers
- **Sheet** - Full-screen overlays
- **Tooltip** - Hover information

#### Forms & Inputs
- **Button** - Primary CTAs and actions
- **Checkbox** - Boolean selections
- **Form** - Form containers and validation
- **Input** - Text input fields
- **Input OTP** - One-time password input
- **Label** - Form labels
- **Radio Group** - Single-select options
- **Select** - Dropdown selections
- **Slider** - Range selectors
- **Switch** - Toggle switches
- **Textarea** - Multi-line text input

#### Data Display
- **Avatar** - User profile images
- **Badge** - Status indicators and tags
- **Card** - Content containers
- **Table** - Data tables
- **Tabs** - Tabbed interfaces
- **Carousel** - Image/content sliders
- **Chart** - Data visualization

#### Feedback
- **Alert** - Inline alerts and messages
- **Progress** - Progress indicators
- **Skeleton** - Loading placeholders
- **Toast** - Notification toasts
- **Sonner** - Advanced toast notifications

#### Utilities
- **Aspect Ratio** - Maintain aspect ratios
- **Collapsible** - Collapsible sections
- **Hover Card** - Hover-triggered content
- **Pagination** - Page navigation
- **Resizable** - Resizable panels
- **Scroll Area** - Styled scrollable areas
- **Separator** - Dividers and separators
- **Toggle** - Toggle buttons
- **Toggle Group** - Toggle button groups

### Component Usage Examples

#### Premium Button
```tsx
import { Button } from '@/components/ui/button';

<Button
  variant="default"
  size="lg"
  className="bg-lii-gold hover:bg-lii-gold-press text-lii-bg font-semibold"
>
  Shop Now
</Button>
```

**Variants:**
- `default` - Primary gold button
- `secondary` - Outlined button
- `ghost` - Transparent button
- `destructive` - Error/delete button
- `outline` - Bordered button
- `link` - Link-styled button

#### Luxury Card
```tsx
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

<Card className="bg-lii-ink border-lii-gold/10 hover:border-lii-gold/20 transition-all">
  <CardHeader>
    <CardTitle className="font-display text-lii-cloud">
      Championship Hoodie
    </CardTitle>
  </CardHeader>
  <CardContent className="text-lii-ash">
    Premium athletic wear for champions
  </CardContent>
</Card>
```

#### Product Badge
```tsx
import { Badge } from '@/components/ui/badge';

<Badge variant="default" className="bg-lii-gold text-lii-bg font-semibold">
  Limited Edition
</Badge>
```

---

## Spacing & Layout

### Spacing Scale

| Size | Rem | Pixels | Tailwind | Usage |
|------|-----|--------|----------|-------|
| **0** | 0 | 0px | `p-0`, `m-0` | No spacing |
| **1** | 0.25rem | 4px | `p-1`, `m-1` | Tight spacing |
| **2** | 0.5rem | 8px | `p-2`, `m-2` | Small spacing |
| **3** | 0.75rem | 12px | `p-3`, `m-3` | Compact spacing |
| **4** | 1rem | 16px | `p-4`, `m-4` | Base spacing |
| **6** | 1.5rem | 24px | `p-6`, `m-6` | Medium spacing |
| **8** | 2rem | 32px | `p-8`, `m-8` | Large spacing |
| **12** | 3rem | 48px | `p-12`, `m-12` | Extra large |
| **16** | 4rem | 64px | `p-16`, `m-16` | Section spacing |
| **20** | 5rem | 80px | `p-20`, `m-20` | Hero spacing |
| **24** | 6rem | 96px | `p-24`, `m-24` | Maximum spacing |

### Layout Patterns

#### Container
```tsx
<div className="container mx-auto px-4 md:px-6 lg:px-8">
  {/* Content */}
</div>
```

#### Grid Layout
```tsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
  {/* Cards */}
</div>
```

#### Flex Layout
```tsx
<div className="flex flex-col md:flex-row items-center justify-between gap-4">
  {/* Items */}
</div>
```

### Border Radius

| Size | Pixels | Tailwind | Usage |
|------|--------|----------|-------|
| **sm** | 2px | `rounded-sm` | Small accents |
| **md** | 6px | `rounded-md` | Form elements |
| **lg** | 8px | `rounded-lg` | Cards, buttons |
| **xl** | 12px | `rounded-xl` | Primary buttons |
| **2xl** | 16px | `rounded-2xl` | Large containers |
| **3xl** | 24px | `rounded-3xl` | Hero sections |
| **full** | 9999px | `rounded-full` | Circular elements |

**Canonical Values:**
- **Card radius**: 16px (`rounded-2xl`)
- **Control radius**: 12px (`rounded-xl`)

---

## Motion & Animation

### Duration Scale

| Speed | Milliseconds | Tailwind | Usage |
|-------|-------------|----------|-------|
| **Fast** | 150ms | `duration-150` | Micro-interactions |
| **Normal** | 300ms | `duration-300` | Standard transitions |
| **Slow** | 500ms | `duration-500` | Deliberate transitions |
| **Slower** | 700ms | `duration-700` | Dramatic effects |

**Canonical Values:**
- **Enter**: 280ms - Entrance animations
- **Micro**: 140ms - Micro-interactions

### Easing Functions

| Name | Value | Tailwind | Usage |
|------|-------|----------|-------|
| **Linear** | linear | `ease-linear` | Constant speed |
| **In** | ease-in | `ease-in` | Accelerating |
| **Out** | ease-out | `ease-out` | Decelerating |
| **In-Out** | ease-in-out | `ease-in-out` | Smooth transitions |
| **Brand** | cubic-bezier(0.16, 1, 0.3, 1) | Custom | Premium feel |

**Brand Easing:**
```css
transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
```

### Common Transitions

#### Color Transitions
```tsx
<button className="bg-lii-gold hover:bg-lii-gold-press transition-colors duration-300">
  Button
</button>
```

#### Transform Transitions
```tsx
<div className="hover:scale-105 hover:-translate-y-1 transition-transform duration-300">
  Card
</div>
```

#### Opacity Transitions
```tsx
<div className="opacity-0 hover:opacity-100 transition-opacity duration-300">
  Content
</div>
```

### Animation Patterns

#### Hover Scale
```tsx
className="transition-transform duration-300 hover:scale-105"
```

#### Hover Lift
```tsx
className="transition-all duration-300 hover:-translate-y-2 hover:shadow-xl"
```

#### Fade In
```tsx
className="animate-in fade-in duration-500"
```

#### Slide In
```tsx
className="animate-in slide-in-from-bottom duration-500"
```

---

## Elevation & Shadows

### Shadow Scale

| Size | Tailwind | Usage |
|------|----------|-------|
| **xs** | `shadow-xs` | Subtle depth |
| **sm** | `shadow-sm` | Slight elevation |
| **md** | `shadow-md` | Card depth |
| **lg** | `shadow-lg` | Elevated cards |
| **xl** | `shadow-xl` | Prominent elevation |
| **2xl** | `shadow-2xl` | Maximum depth |

**Custom Shadows:**
```css
/* Base - Default card shadow */
box-shadow: 0 8px 24px rgba(193, 160, 96, 0.12);

/* Hover - Elevated card */
box-shadow: 0 12px 32px rgba(193, 160, 96, 0.16);

/* Active - Pressed state */
box-shadow: 0 4px 16px rgba(193, 160, 96, 0.08);
```

### Elevation Patterns

#### Flat Surface (Z-index: 0)
```tsx
className="shadow-none"
```

#### Card Surface (Z-index: 1)
```tsx
className="shadow-base"
```

#### Elevated Card (Z-index: 2)
```tsx
className="shadow-hover"
```

#### Modal/Dialog (Z-index: 50)
```tsx
className="shadow-2xl"
```

---

## Design Patterns

### Product Card Pattern
```tsx
<Card className="group bg-lii-ink border-lii-gold/10 hover:border-lii-gold/20 transition-all duration-300 hover:-translate-y-2 hover:shadow-xl">
  <div className="aspect-square overflow-hidden rounded-t-2xl">
    <img
      src="/product.jpg"
      alt="Product"
      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
    />
  </div>
  <CardContent className="p-6">
    <Badge className="mb-3 bg-lii-gold text-lii-bg">Limited Edition</Badge>
    <h3 className="font-display text-2xl font-semibold text-lii-cloud mb-2">
      Championship Hoodie
    </h3>
    <p className="text-lii-ash mb-4">
      Premium athletic wear for champions
    </p>
    <div className="flex items-center justify-between">
      <span className="font-display text-3xl font-bold text-lii-gold">
        $89.99
      </span>
      <Button className="bg-lii-gold hover:bg-lii-gold-press text-lii-bg">
        Add to Cart
      </Button>
    </div>
  </CardContent>
</Card>
```

### Hero Section Pattern
```tsx
<section className="relative min-h-screen flex items-center justify-center bg-lii-bg overflow-hidden">
  {/* Background gradient */}
  <div className="absolute inset-0 bg-gradient-to-br from-lii-gold/5 to-transparent" />

  {/* Content */}
  <div className="container relative z-10 text-center">
    <h1 className="font-display text-6xl md:text-8xl font-bold tracking-tight text-lii-cloud mb-6 animate-in fade-in slide-in-from-bottom duration-700">
      Live It Iconic
    </h1>
    <p className="font-ui text-lg md:text-xl text-lii-ash max-w-2xl mx-auto mb-8 animate-in fade-in slide-in-from-bottom duration-700 delay-150">
      Premium automotive lifestyle merchandise for those who live iconic
    </p>
    <Button
      size="lg"
      className="bg-lii-gold hover:bg-lii-gold-press text-lii-bg font-semibold animate-in fade-in slide-in-from-bottom duration-700 delay-300"
    >
      Explore Collection
    </Button>
  </div>
</section>
```

### Navigation Pattern
```tsx
<nav className="sticky top-0 z-50 bg-lii-bg/95 backdrop-blur-md border-b border-lii-gold/10">
  <div className="container mx-auto px-4 md:px-6">
    <div className="flex items-center justify-between h-20">
      {/* Logo */}
      <IconicLogo variant="primary" size="md" />

      {/* Navigation */}
      <div className="hidden md:flex items-center gap-8">
        <a href="/shop" className="font-ui text-lii-cloud hover:text-lii-gold transition-colors">
          Shop
        </a>
        <a href="/collections" className="font-ui text-lii-cloud hover:text-lii-gold transition-colors">
          Collections
        </a>
        <a href="/about" className="font-ui text-lii-cloud hover:text-lii-gold transition-colors">
          About
        </a>
      </div>

      {/* CTA */}
      <Button variant="default" className="bg-lii-gold hover:bg-lii-gold-press text-lii-bg">
        Cart (0)
      </Button>
    </div>
  </div>
</nav>
```

---

## Template Guidelines

### Page Templates

#### 1. Homepage Template
**Structure:**
- Hero section with primary CTA
- Featured products carousel
- Collection highlights
- Brand story section
- Email capture
- Footer

**Key Elements:**
- Large display typography
- Gold accents
- Product showcases
- Social proof

#### 2. Product Listing Page
**Structure:**
- Filter sidebar
- Product grid (3-4 columns)
- Pagination
- Sort controls

**Key Elements:**
- Product cards with hover states
- Clear CTAs
- Price display
- Badge indicators

#### 3. Product Detail Page
**Structure:**
- Image gallery (left)
- Product info (right)
- Specifications accordion
- Related products
- Reviews section

**Key Elements:**
- Large product images
- Add to cart CTA
- Size/variant selector
- Detailed descriptions

#### 4. Checkout Page
**Structure:**
- Progress indicator
- Form sections
- Order summary (sticky)
- Payment integration

**Key Elements:**
- Clear form labels
- Validation messages
- Secure payment indicators
- Order total breakdown

### Component Reusability

**DO:**
- ✅ Use consistent spacing (4, 6, 8 multiples)
- ✅ Apply hover states to interactive elements
- ✅ Use semantic HTML elements
- ✅ Follow mobile-first responsive design
- ✅ Implement loading states

**DON'T:**
- ❌ Create one-off components
- ❌ Hard-code colors (use tokens)
- ❌ Neglect accessibility
- ❌ Ignore responsive breakpoints
- ❌ Skip error states

---

## Accessibility Standards

### WCAG 2.1 AA Compliance

#### Color Contrast
**Minimum Ratios:**
- **Normal text**: 4.5:1
- **Large text** (18pt+): 3:1
- **UI components**: 3:1

**Compliant Combinations:**
- ✅ Cloud (#E6E9EF) on Carbon Black (#0B0B0C) - 13.8:1
- ✅ Ash (#8C93A3) on Carbon Black - 7.2:1
- ✅ Gold (#d4af37) on Carbon Black - 6.5:1
- ✅ Carbon Black on Gold - 6.5:1

#### Keyboard Navigation
- All interactive elements must be keyboard accessible
- Visible focus states required
- Logical tab order
- Skip navigation links

#### Screen Reader Support
- Semantic HTML structure
- ARIA labels for icons
- Alt text for images
- Descriptive link text

#### Focus Indicators
```tsx
className="focus:outline-none focus:ring-2 focus:ring-lii-gold focus:ring-offset-2 focus:ring-offset-lii-bg"
```

### Accessibility Checklist

- [ ] All images have alt text
- [ ] Forms have proper labels
- [ ] Buttons have descriptive text
- [ ] Color is not the only indicator
- [ ] Text is resizable to 200%
- [ ] Focus states are visible
- [ ] Semantic HTML is used
- [ ] ARIA labels are provided where needed
- [ ] Keyboard navigation works throughout
- [ ] Screen reader tested

---

## Component Review & Recommendations

### Components to Keep (Radix UI)

**Essential Components:**
- ✅ Button, Card, Dialog, Dropdown Menu, Form, Input, Select, Tabs, Toast

**Recommended Removals:**
Based on usage analysis, consider removing unused Radix components to reduce bundle size:

- ⚠️ **Menubar** - If not used in navigation
- ⚠️ **Command** - If no command palette
- ⚠️ **Calendar** - If no date pickers
- ⚠️ **Chart** - If no data visualization
- ⚠️ **Input OTP** - If no 2FA
- ⚠️ **Resizable** - If no resizable panels
- ⚠️ **Toggle Group** - If not used

**Action:** Run tree-shaking analysis to identify unused components

---

## Design System Evolution

### Version History

#### v1.0.0 (Current)
- Established canonical color palette
- Defined typography scale
- Created logo collection
- Implemented Radix UI components
- Documented design tokens

#### Planned Improvements

**v1.1.0:**
- Dark mode variants
- Animation library
- Icon system standardization
- Component usage analytics

**v1.2.0:**
- Mobile-specific patterns
- Micro-interactions library
- Performance optimizations
- Accessibility audit

---

## Resources & Tools

### Design Tools
- **Figma**: Design files and prototypes
- **Tailwind CSS**: Utility-first styling
- **Radix UI**: Accessible components
- **Lucide Icons**: Icon library

### Development Tools
- **Storybook**: Component playground (planned)
- **Chromatic**: Visual regression testing (planned)
- **axe DevTools**: Accessibility testing

### Documentation
- [API Documentation](./API.md)
- [Architecture Guide](./ARCHITECTURE.md)
- [Project Reference](./PROJECT_REFERENCE.md)

---

**Maintained by:** Design Team & Engineering Team
**Last Review:** 2025-11-12
**Next Review:** 2026-01-12

---

## Quick Reference Card

### Brand Colors
```
Gold: #d4af37
Carbon Black: #0B0B0C
Cloud: #E6E9EF
Ash: #8C93A3
```

### Typography
```
Display: Playfair Display
UI: Inter Variable
Mono: JetBrains Mono
```

### Spacing
```
Small: 4px, 8px, 12px
Medium: 16px, 24px, 32px
Large: 48px, 64px, 96px
```

### Border Radius
```
Card: 16px (rounded-2xl)
Control: 12px (rounded-xl)
```

### Transitions
```
Duration: 300ms
Easing: cubic-bezier(0.16, 1, 0.3, 1)
```
