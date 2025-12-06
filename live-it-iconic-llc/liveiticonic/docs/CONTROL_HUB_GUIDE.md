# Live It Iconic - AI Control Hub Guide

**Purpose:** Canonical instructions for AI assistants working on Live It Iconic  
**Last Updated:** 2025-11-08  
**Version:** 1.0.0

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Design System](#design-system)
4. [Type System](#type-system)
5. [Backend](#backend)
6. [Development Workflow](#development-workflow)
7. [Brand Guidelines](#brand-guidelines)
8. [Common Patterns](#common-patterns)
9. [Security](#security)
10. [Resources](#resources)

---

## 📖 Project Overview

**Live It Iconic** is a luxury automotive lifestyle apparel brand.

### Core Identity
- **Positioning:** Caribbean-inspired, performance-oriented lifestyle brand for automotive enthusiasts
- **Aesthetic:** Minimalist, elegant, disciplined luxury
- **Voice:** Confident, restrained, precise. Short sentences. No hype or exclamation marks.
- **Tagline:** *Live it iconic.*

### Technology Stack
- **Frontend:** React + TypeScript + Vite + Tailwind CSS
- **Backend:** Lovable Cloud (Supabase)
- **Architecture:** Feature-first monorepo (future)
- **Design:** Token-based design system

---

## 🏗️ Architecture

### Current Structure (Single App)
```
src/
├── components/        # Shared UI components
│   ├── ui/           # shadcn components
│   ├── logo/         # Brand logos
│   ├── Navigation.tsx
│   └── Footer.tsx
├── pages/            # Route pages
│   ├── Index.tsx     # Homepage
│   ├── About.tsx
│   ├── Collections.tsx
│   ├── BrandAssets.tsx
│   └── Lifestyle.tsx
├── theme/
│   └── tokens.ts     # Design tokens (CANONICAL)
├── hooks/            # Custom React hooks
├── lib/              # Utilities
└── index.css         # CSS variables + Tailwind
```

### Future Monorepo Structure
```
apps/
├── website/          # Marketing (landing, lifestyle, about)
├── shop/             # E-commerce (PLP, PDP, cart, checkout)
└── admin/            # Inventory & order management

packages/
├── ui/               # Shared components
├── types/            # TypeScript definitions
├── design-tokens/    # Design system
└── utils/            # Shared utilities

supabase/             # Backend (at root)
```

### Feature-First Organization

When building new features, use this pattern:

```
src/features/cart/
├── components/
│   ├── CartDrawer.tsx
│   ├── CartItem.tsx
│   └── CartSummary.tsx
├── hooks/
│   └── useCart.ts
├── api/
│   └── cartApi.ts
├── types.ts
└── index.ts
```

**Benefits:**
- Related code stays together
- Easy to find and modify
- Clear boundaries between features

---

## 🎨 Design System

### Token Hierarchy

```
src/theme/tokens.ts (Source of Truth)
  ↓
tailwind.config.ts (extends tokens)
  ↓
src/index.css (CSS variables)
  ↓
Components (use semantic classes)
```

### Color Palette (CANONICAL)

**From `src/theme/tokens.ts`:**

```typescript
export const tokens = {
  colors: {
    // Neutral Palette
    bg: '#0B0B0C',        // Primary background
    ink: '#14161A',        // Secondary background
    charcoal: '#2A2D33',   // Borders
    graphite: '#3A4048',   // Component backgrounds
    ash: '#8C93A3',        // Body text
    cloud: '#E6E9EF',      // Headings
    
    // Brand Gold
    gold: '#C1A060',       // Primary accent (Championship Gold)
    goldPress: '#AE8D55',  // Gold pressed/hover state
    goldHover: '#e0c050',  // Gold lighter hover
    
    // Semantic
    signalRed: '#E03A2F',  // Errors, warnings
    success: '#10B981',    // Success states
    warning: '#F59E0B',    // Warning states
  }
};
```

### Usage Rules (CRITICAL)

❌ **NEVER:**
```tsx
<div className="bg-yellow-600 text-white">
<p className="text-gray-400">
<button className="bg-black border-white">
```

✅ **ALWAYS:**
```tsx
<div className="bg-lii-gold text-lii-bg">
<p className="text-lii-ash">
<button className="bg-lii-ink border-lii-charcoal">
```

✅ **PREFERRED (semantic):**
```tsx
<div className="bg-accent text-accent-foreground">
<p className="text-muted-foreground">
<button className="bg-card border-border">
```

### Typography

**From `src/theme/tokens.ts`:**

```typescript
typography: {
  fonts: {
    display: ['Playfair Display', 'serif'],  // H1-H2
    ui: ['Inter Variable', 'sans-serif'],     // Body, UI
    mono: ['JetBrains Mono', 'monospace'],    // Code
  }
}
```

**Usage:**
```tsx
<h1 className="font-display text-6xl tracking-tight">Live it iconic.</h1>
<p className="font-ui text-base text-lii-ash">Statement pieces for bold days.</p>
```

**Scale (Mobile → Desktop):**
- H1: 40px/44px → 56px/60px (tracking −1%)
- H2: 28px/32px → 40px/44px (tracking −1%)
- H3: 22px/28px → 28px/32px (Inter Semibold)
- Body: 16–18px / 24–28px (Inter Regular)
- Caption: 12px/16px (Inter Medium, +2% letter-spacing)

### Motion

**From `src/theme/tokens.ts`:**

```typescript
motion: {
  duration: {
    enter: 280,   // ms - Entrance animations
    micro: 140,   // ms - Micro-interactions
  },
  ease: 'cubic-bezier(0.16, 1, 0.3, 1)',  // Brand easing
}
```

**Usage:**
- Entrances: 240–320 ms, ease-out-quart
- Micro-interactions: 120–160 ms
- Parallax caps: 6–10 px
- Transforms + opacity only (no animating blurs/shadows)

### Component Tokens

**Buttons:**
```typescript
// Primary
bg-lii-gold text-lii-bg
border-lii-gold rounded-control h-12 px-8
font-semibold transition-all duration-micro

// Secondary (Outline)
bg-transparent text-lii-gold
border-1.5 border-lii-gold rounded-control h-12 px-8

// Ghost
bg-transparent text-lii-cloud
hover:bg-lii-graphite/20
```

**Cards:**
```typescript
bg-lii-ink/30 rounded-card border border-lii-charcoal/30
shadow-base hover:shadow-hover
transition-all duration-enter
```

---

## 🗂️ Type System (Source of Truth)

### Product Types

**Location:** `src/types/product.ts` (or `packages/types/src/product.ts` in monorepo)

```typescript
export interface Product {
  id: string;
  sku: string;
  title: string;
  subtitle?: string;
  description: string;
  price: number;
  compareAtPrice?: number;
  images: string[];
  colors: ColorVariant[];
  sizes: SizeVariant[];
  materials: string[];
  careInstructions: string;
  collection: 'street' | 'performance' | 'essentials';
  tags: string[];
  inventory: number;
}

export interface Collection {
  slug: string;
  title: string;
  description: string;
  heroImage: string;
  products: Product[];
  filters: Filter[];
}
```

### Order Types

```typescript
export interface Order {
  id: string;
  customerId: string;
  items: CartItem[];
  subtotal: number;
  shipping: number;
  tax: number;
  total: number;
  shippingAddress: Address;
  status: 'pending' | 'confirmed' | 'shipped' | 'delivered';
  createdAt: Date;
}

export interface CartItem {
  productId: string;
  variantId: string;
  quantity: number;
  price: number;
}
```

**Rules:**
- NEVER duplicate type definitions
- Always import from canonical location
- Update types FIRST, then use everywhere

---

## 🔐 Backend (Lovable Cloud / Supabase)

### Edge Functions

**Location:** `/supabase/functions/[function-name]/index.ts`

**Template:**
```typescript
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL')!,
      Deno.env.get('SUPABASE_ANON_KEY')!
    );

    // Your logic here
    const { data, error } = await supabase
      .from('products')
      .select('*');

    if (error) throw error;

    return new Response(JSON.stringify({ data }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Function error:', error);
    return new Response(
      JSON.stringify({ error: 'An error occurred' }),
      { status: 500, headers: corsHeaders }
    );
  }
});
```

**Guidelines:**
- Always include CORS headers
- Log errors server-side (console.error)
- Return generic error messages to clients
- Use `supabase.from()` methods, NOT raw SQL
- Validate all inputs

### Database Schema

**Tables:**
- `products` - Product catalog
- `collections` - Product collections
- `orders` - Customer orders
- `customers` - Customer profiles
- `inventory` - Stock levels

**RLS Policies:** ALWAYS enable on user-facing tables

```sql
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own orders"
ON orders FOR SELECT
USING (auth.uid() = customer_id);
```

---

## 🚀 Development Workflow

### Creating a New Feature

1. **Plan structure:**
   ```bash
   mkdir -p src/features/wishlist
   ```

2. **Create files:**
   ```
   wishlist/
   ├── components/
   │   └── WishlistButton.tsx
   ├── hooks/
   │   └── useWishlist.ts
   ├── types.ts
   └── index.ts
   ```

3. **If reusable, extract to shared:**
   - Move to `src/components/` or `packages/ui/`
   - Export from barrel file
   - Import where needed

### Adding a Shared Component

1. **Create component:**
   ```typescript
   // src/components/ProductCard.tsx
   export function ProductCard({ product }: { product: Product }) {
     return (
       <div className="bg-lii-ink rounded-card border border-lii-charcoal p-6">
         <img src={product.images[0]} alt={product.title} />
         <h3 className="font-ui text-xl text-lii-cloud">{product.title}</h3>
         <p className="text-lii-gold">${product.price}</p>
       </div>
     );
   }
   ```

2. **Export:**
   ```typescript
   // src/components/index.ts
   export { ProductCard } from './ProductCard';
   ```

3. **Use:**
   ```typescript
   // src/pages/Collections.tsx
   import { ProductCard } from '@/components';
   ```

---

## 🎨 Brand Guidelines

### Voice & Tone

**DO:**
- Confident, restrained, precise
- Short sentences
- Factual, specific copy
- Lead with value (materials, fit, care)

**DON'T:**
- No hype or exclamation marks
- No fabricated metrics (fake reviews, sales numbers)
- No "concept/prototype/TBD" language
- No ALL-CAPS paragraphs

### Copy Examples

**Hero:**
```
H1: Live it iconic.
Sub: Statement pieces for bold days.
CTA: Shop the drop | Explore lifestyle
```

**Product Description:**
```
Premium cotton blend. Tailored fit. 
Designed for motion; refined for nights out.
Machine wash cold. Tumble dry low.
```

**About (Lead, ≤80 words):**
```
We cut apparel with the discipline of performance machines—
clean lines, durable fabrics, precise fits. 
Designed for motion; refined for nights out.
```

### Logo Usage

**Available Logos:**
- `LiveItIconicWordmark` - Horizontal/stacked wordmark (primary)
- `IconicDiamondMark` - Mark-only (favicon, social)
- `AutomotiveCrestLogo` - Shield/crest variant
- `MinimalistRoadmark` - Road/motion symbol

**Rules:**
- Clearspace = cap height of "I"
- Min digital size (mark) = 24 px
- Header/footer: cloud on ink
- Gold for accents/hover only, never body text

### Color Ratio
- 80% neutrals (bg, ink, charcoal, graphite, ash, cloud)
- 15% cloud/ash
- ≤5% gold/red

**Never:** Small gold text on dark backgrounds (contrast risk)

---

## 🧩 Common Patterns

### Product Card

```typescript
export function ProductCard({ product }: { product: Product }) {
  return (
    <div className="group bg-lii-ink/30 rounded-card border border-lii-charcoal/30 
                    hover:border-lii-gold/50 transition-all duration-enter">
      <div className="aspect-square overflow-hidden rounded-t-card">
        <img 
          src={product.images[0]} 
          alt={`${product.title} - ${product.materials.join(', ')}`}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform"
        />
      </div>
      <div className="p-4">
        <h3 className="font-ui font-semibold text-lii-cloud">{product.title}</h3>
        <p className="text-sm text-lii-ash mt-1">{product.subtitle}</p>
        <div className="flex items-center justify-between mt-4">
          <span className="font-numeric text-lii-gold">${product.price}</span>
          <button className="bg-lii-gold text-lii-bg px-4 py-2 rounded-control 
                             hover:bg-lii-gold-press transition-colors">
            Add to Cart
          </button>
        </div>
      </div>
    </div>
  );
}
```

### Button Component

```typescript
import { cva } from 'class-variance-authority';

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-control font-semibold transition-all duration-micro',
  {
    variants: {
      variant: {
        primary: 'bg-lii-gold text-lii-bg hover:bg-lii-gold-press',
        secondary: 'bg-transparent text-lii-gold border-1.5 border-lii-gold hover:bg-lii-gold/10',
        ghost: 'bg-transparent text-lii-cloud hover:bg-lii-graphite/20',
      },
      size: {
        sm: 'h-10 px-4 text-sm',
        md: 'h-12 px-6 text-base',
        lg: 'h-14 px-8 text-lg',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
    },
  }
);
```

---

## 🔒 Security

### RLS Policies
```sql
-- Enable RLS
ALTER TABLE products ENABLE ROW LEVEL SECURITY;

-- Public read access
CREATE POLICY "Products are viewable by everyone"
ON products FOR SELECT
USING (true);

-- Only admins can modify
CREATE POLICY "Admins can manage products"
ON products FOR ALL
USING (auth.jwt() ->> 'role' = 'admin');
```

### Edge Functions
- NEVER expose internal errors
- Log sensitive details server-side only
- Generic client messages: "An error occurred"
- Validate ALL user input
- Rate limit endpoints

---

## 📞 Resources

### Documentation
- **Monorepo Structure:** `/MONOREPO_STRUCTURE.md`
- **Control Hub:** `/docs/CONTROL_HUB_GUIDE.md` (this file)
- **Knowledge Transfer:** `/docs/LOVABLE_KNOWLEDGE_TRANSFER.md`
- **Brand Guidelines:** `/platform/docs/Brand-Guidelines.tex`
- **Design System:** `/src/theme/tokens.ts`

### External
- [Lovable Docs](https://docs.lovable.dev/)
- [Supabase Docs](https://supabase.com/docs)
- [Tailwind CSS](https://tailwindcss.com/)
- [shadcn/ui](https://ui.shadcn.com/)

---

## ✅ Checklist for New AI Sessions

When starting work on this project, verify:

- [ ] Understand brand identity: luxury, minimalist, automotive
- [ ] Know color palette: bg/ink/charcoal/graphite/ash/cloud/gold
- [ ] Use design tokens, never arbitrary colors
- [ ] Follow feature-first organization
- [ ] Enable RLS on user-facing tables
- [ ] Include CORS in edge functions
- [ ] Use confident, restrained copy
- [ ] Logo clearspace = cap height of "I"
- [ ] Never use gold for body text on dark
- [ ] Understand product/collection/order types

---

**Last Updated:** 2025-11-08  
**Version:** 1.0.0  
**Maintained By:** Technical Lead & AI Assistant
