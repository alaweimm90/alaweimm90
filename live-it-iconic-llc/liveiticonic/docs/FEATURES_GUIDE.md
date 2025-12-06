# Live It Iconic - Features Guide & FAQ

**Version:** 1.0.0
**Last Updated:** 2025-11-12
**Purpose:** Comprehensive guide to platform features, components, and usage

---

## Table of Contents

1. [Platform Overview](#platform-overview)
2. [Available Features](#available-features)
3. [Apparel & Products](#apparel--products)
4. [YouTube & Media Integration](#youtube--media-integration)
5. [Component Showcases](#component-showcases)
6. [Page Templates](#page-templates)
7. [Logo & Branding Assets](#logo--branding-assets)
8. [Filters & Animations](#filters--animations)
9. [Tailwind Components Review](#tailwind-components-review)
10. [Template Best Practices](#template-best-practices)
11. [FAQ](#faq)

---

## Platform Overview

**Live It Iconic** is a luxury automotive lifestyle e-commerce platform featuring:

### Core Capabilities
- **E-Commerce**: Full shopping cart, checkout, and payment processing (Stripe)
- **Product Catalog**: Premium apparel and accessories
- **AI Launch Platform**: 26-agent system for product launches
- **Brand Storytelling**: Multimedia content showcases
- **Responsive Design**: Mobile-first, accessible interface

### Technology Stack
- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS + Radix UI components
- **Backend**: Supabase (Auth, Database, Storage)
- **Payments**: Stripe integration
- **Animations**: CSS animations + Canvas animations

---

## Available Features

### ✅ Implemented Features

#### 1. Homepage (src/pages/Index.tsx)
**Sections:**
- Hero section with premium branding
- Product showcase (featured items)
- Minimal showcase (collection highlights)
- Brand manifesto
- Bird logo showcase (seasonal)
- Supercar gallery
- Lifestyle teaser
- Podcast showcase (YouTube integration ready)
- Email capture

**Access:** Navigate to `/` or homepage

#### 2. Shop Page (src/pages/Shop.tsx)
**Features:**
- Product grid with filters
- Category navigation
- Search functionality
- Add to cart
- Product detail views

**Access:** Navigate to `/shop`

#### 3. Product Detail Page (src/pages/ProductDetail.tsx)
**Features:**
- Image gallery
- Product specifications
- Size/variant selector
- Add to cart
- Related products
- Reviews section

**Access:** Navigate to `/product/:id`

#### 4. Checkout Page (src/pages/Checkout.tsx)
**Features:**
- Multi-step checkout flow
- Shipping address form
- Payment integration (Stripe)
- Order summary
- Order confirmation

**Access:** Navigate to `/checkout`

#### 5. Launch Platform (src/launch-platform/)
**Features:**
- 26 specialized AI agents
- Product launch orchestration
- Campaign management
- Analytics and optimization
- Event-driven architecture

**Access:** Programmatic API (see docs/API.md)

---

## Apparel & Products

### Product Categories

#### 1. Apparel Collection
**Available Products:**

##### Black T-Shirt
**Location:** `src/assets/products/BlackTShirtMockup.tsx`
**Features:**
- Premium cotton blend
- Minimalist Caribbean bird logo
- Championship gold accent
- Sizes: XS, S, M, L, XL, XXL

**Usage:**
```tsx
import { BlackTShirtMockup } from '@/assets/products';

<BlackTShirtMockup className="w-full h-auto" />
```

##### Black Hoodie
**Location:** `src/assets/products/BlackHoodieMockup.tsx`
**Features:**
- Premium athletic hoodie
- Embroidered logo
- Championship gold details
- Sizes: S, M, L, XL, XXL

**Usage:**
```tsx
import { BlackHoodieMockup } from '@/assets/products';

<BlackHoodieMockup className="w-full h-auto" />
```

##### Black Cap
**Location:** `src/assets/products/BlackCapMockup.tsx`
**Features:**
- Structured athletic cap
- Embroidered front logo
- Adjustable strap
- One size fits all

**Usage:**
```tsx
import { BlackCapMockup } from '@/assets/products';

<BlackCapMockup className="w-full h-auto" />
```

### How to Access Apparel Examples

#### Method 1: Homepage Showcase
1. Navigate to the homepage (`/`)
2. Scroll to the **Product Showcase** section
3. View featured apparel with interactive cards

#### Method 2: Shop Page
1. Navigate to `/shop`
2. Filter by category: **Apparel**
3. Browse all available products
4. Click any product for detailed view

#### Method 3: Direct Product Assets
```tsx
// Import all product mockups
import {
  BlackTShirtMockup,
  BlackHoodieMockup,
  BlackCapMockup
} from '@/assets/products';

// Use in your component
<div className="grid grid-cols-3 gap-6">
  <BlackTShirtMockup />
  <BlackHoodieMockup />
  <BlackCapMockup />
</div>
```

### Product Specifications

#### T-Shirt Specs
```typescript
{
  name: "Championship T-Shirt",
  price: 49.99,
  category: "apparel",
  material: "Premium Cotton Blend",
  fit: "Athletic Fit",
  features: [
    "Breathable fabric",
    "Embroidered logo",
    "Reinforced seams",
    "Machine washable"
  ],
  sizes: ["XS", "S", "M", "L", "XL", "XXL"],
  colors: ["Carbon Black"]
}
```

#### Hoodie Specs
```typescript
{
  name: "Performance Hoodie",
  price: 89.99,
  category: "apparel",
  material: "Premium Cotton/Polyester Blend",
  fit: "Regular Fit",
  features: [
    "Fleece-lined interior",
    "Kangaroo pocket",
    "Adjustable hood",
    "Ribbed cuffs and hem"
  ],
  sizes: ["S", "M", "L", "XL", "XXL"],
  colors: ["Carbon Black"]
}
```

---

## YouTube & Media Integration

### Podcast Showcase Component

**Location:** `src/components/PodcastShowcase.tsx`

#### Status: ✅ Implemented & Ready

**Features:**
- Canvas-based animated background
- YouTube icon integration (Lucide React)
- Floating orb animations
- Responsive layout
- Four perspective cards:
  - **The Athlete**: Peak Performance
  - **The Entrepreneur**: Strategic Vision
  - **The Creator**: Artistic Excellence
  - **The Dreamer**: Aspirational Living

#### Does It Work?
**Yes**, the component is fully functional and displays on the homepage.

**Current Implementation:**
- ✅ Visual design complete
- ✅ Animations working
- ✅ YouTube icon present
- ⚠️ YouTube embed placeholder (needs actual channel URL)

#### How to Add YouTube Channel

**Step 1: Add YouTube Video ID**

Edit `src/components/PodcastShowcase.tsx`:

```tsx
// Add this constant at the top of the file
const YOUTUBE_VIDEO_ID = "YOUR_VIDEO_ID"; // e.g., "dQw4w9WgXcQ"
const YOUTUBE_CHANNEL_URL = "https://youtube.com/@yourusername";

// Add iframe embed in the component
<div className="aspect-video rounded-2xl overflow-hidden">
  <iframe
    width="100%"
    height="100%"
    src={`https://www.youtube.com/embed/${YOUTUBE_VIDEO_ID}`}
    title="Live It Iconic Podcast"
    frameBorder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen
    className="w-full h-full"
  />
</div>
```

**Step 2: Add CTA Button**

```tsx
<Button
  size="lg"
  className="bg-lii-gold hover:bg-lii-gold-press text-lii-bg font-semibold"
  onClick={() => window.open(YOUTUBE_CHANNEL_URL, '_blank')}
>
  <Youtube className="mr-2 h-5 w-5" />
  Subscribe to Channel
</Button>
```

**Step 3: Test**

1. Run dev server: `npm run dev`
2. Navigate to homepage
3. Scroll to "Podcast Showcase" section
4. Video should be embedded and playable

### Social Media Integration

**Location:** `src/components/Footer.tsx`

**Supported Platforms:**
- Instagram
- Twitter/X
- Facebook
- LinkedIn
- YouTube

**Configuration:**

Edit footer social links:
```tsx
<a
  href="https://youtube.com/@liveiticonic"
  target="_blank"
  rel="noopener noreferrer"
  className="text-lii-ash hover:text-lii-gold transition-colors"
>
  <Youtube className="h-5 w-5" />
</a>
```

---

## Component Showcases

### Available Showcase Components

#### 1. Product Showcase
**Location:** `src/components/ProductShowcase.tsx`
**Purpose:** Display featured products with animations
**Length:** 283 lines (⚠️ Needs refactoring)

**Features:**
- Product grid with hover effects
- Add to cart functionality
- Price display
- Badge indicators (Limited Edition, New, etc.)
- Responsive columns (1-3 based on screen size)

**Access:** Visible on homepage

#### 2. Bird Logo Showcase
**Location:** `src/components/BirdLogoShowcase.tsx`
**Purpose:** Seasonal bird logo carousel

**Variants:**
- `full` - Grid display
- `compact` - Condensed view
- `carousel` - Scrollable carousel (default)

**Usage:**
```tsx
<BirdLogoShowcase variant="carousel" className="py-12" />
```

**Features:**
- 4 seasonal bird designs
- Automatic season detection
- Smooth carousel animations
- Characteristics display

#### 3. Minimal Showcase
**Location:** `src/components/MinimalShowcase.tsx`
**Purpose:** Clean, minimalist product highlight

**Usage:**
```tsx
<MinimalShowcase />
```

#### 4. Supercar Gallery
**Location:** `src/components/SupercarGallery.tsx`
**Purpose:** Automotive lifestyle imagery

**Features:**
- High-quality car imagery
- Luxury brand association
- Parallax effects
- Call-to-action

#### 5. Icon Showcase
**Location:** `src/components/IconShowcase.tsx`
**Length:** 277 lines (⚠️ Needs refactoring)
**Purpose:** Display all 16 icon logos

**Features:**
- Grid layout
- Interactive hover states
- Size variations
- Color customization

### Creating Custom Showcases

**Template Pattern:**

```tsx
import { Card } from '@/components/ui/card';

export const CustomShowcase = () => {
  return (
    <section className="py-20 bg-lii-bg">
      <div className="container mx-auto px-6">
        <h2 className="font-display text-4xl font-bold text-lii-cloud text-center mb-12">
          Showcase Title
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Showcase items */}
        </div>
      </div>
    </section>
  );
};
```

---

## Page Templates

### Available Page Templates

#### 1. Homepage Template
**File:** `src/pages/Index.tsx`

**Sections:**
1. Hero (full-screen landing)
2. ProductShowcase (featured products)
3. MinimalShowcase (collection highlight)
4. BrandManifesto (brand story)
5. BirdLogoShowcase (seasonal branding)
6. SupercarGallery (lifestyle imagery)
7. LifestyleTeaser (lifestyle content)
8. PodcastShowcase (media/YouTube)
9. EmailCapture (newsletter signup)

**Usage:** Default landing page at `/`

#### 2. Shop Template
**File:** `src/pages/Shop.tsx`

**Sections:**
- Filter sidebar (category, price, size)
- Product grid (3-4 columns)
- Pagination controls
- Sort dropdown

**Usage:** Product listing at `/shop`

#### 3. Product Detail Template
**File:** `src/pages/ProductDetail.tsx`

**Layout:**
- Left: Image gallery
- Right: Product info, CTA
- Below: Specifications, reviews

**Usage:** Individual product at `/product/:id`

#### 4. Checkout Template
**File:** `src/pages/Checkout.tsx`

**Steps:**
1. Cart review
2. Shipping information
3. Payment details
4. Order confirmation

**Usage:** Checkout flow at `/checkout`

#### 5. About Template
**File:** `src/pages/About.tsx`

**Sections:**
- Brand story
- Team/founders
- Mission & values
- Timeline

**Usage:** About page at `/about`

#### 6. Contact Template
**File:** `src/pages/Contact.tsx`

**Features:**
- Contact form
- Social links
- Business hours
- Location map (optional)

**Usage:** Contact page at `/contact`

### Template Customization

**Steps to Create New Page:**

1. **Create file:** `src/pages/NewPage.tsx`

```tsx
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import SEO from '@/components/SEO';

const NewPage = () => {
  return (
    <>
      <SEO
        title="Page Title"
        description="Page description"
        canonical="/new-page"
      />
      <div className="min-h-screen bg-lii-bg">
        <Navigation />
        <main className="container mx-auto px-6 py-20">
          {/* Your content */}
        </main>
        <Footer />
      </div>
    </>
  );
};

export default NewPage;
```

2. **Add route:** `src/App.tsx`

```tsx
import NewPage from '@/pages/NewPage';

// In Routes component
<Route path="/new-page" element={<NewPage />} />
```

3. **Add navigation link:** `src/components/Navigation.tsx`

```tsx
<a href="/new-page" className="text-lii-cloud hover:text-lii-gold">
  New Page
</a>
```

---

## Logo & Branding Assets

### Complete Logo Inventory

#### Main Brand Logos (4)
1. **IconicLogo** - Primary brand logo
2. **ElegantLogo** - Luxury serif treatment
3. **PremiumLogo** - Contemporary style
4. **AutomotiveCrestLogo** - Automotive-inspired crest

#### Icon Logos (16)
1. **ApexLogo** - Performance
2. **AuroraLogo** - Premium
3. **DiamondLogo** - Luxury
4. **ElevationLogo** - Aspirational
5. **InfinityLogo** - Timeless
6. **NexusLogo** - Connected
7. **NovaLogo** - New launches
8. **PhoenixLogo** - Special editions
9. **PrestigeLogo** - Refined
10. **PulseLogo** - Athletic
11. **RadiantLogo** - Premium highlights
12. **SummitLogo** - Championship
13. **VelocityLogo** - Racing
14. **VortexLogo** - Dynamic
15. **ZenithLogo** - Top-tier

#### Bird Logos (4 Seasonal)
1. **ElegantFlamingo** - Jan-Mar (Grace)
2. **MajesticPelican** - Apr-Jun (Strength)
3. **CaribbeanFrigateBird** - Jul-Sep (Power)
4. **TropicalTanager** - Oct-Dec (Vibrancy)

#### Brandmarks (15)
1. **ArtDecoStyle** - Vintage luxury
2. **ArchitecturalBlock** - Modern structured
3. **BoldStatement** - Strong impact
4. **ClassicEmblem** - Traditional refined
5. **DiagonalDynamic** - Energetic modern
6. **FuturisticType** - Cutting-edge
7. **GeometricInitials** - Clean minimalist
8. **HandwrittenSignature** - Personal authentic
9. **InitialsCircle** - Contained elegant
10. **LuxurySerif** - Classic sophisticated
11. **MinimalistStack** - Simple modern
12. **ModernWordmark** - Contemporary bold
13. **MonogramShield** - Protected prestigious
14. **ScriptElegance** - Flowing refined
15. **TechMonospace** - Technical precise

### Logo Usage Guide

**See comprehensive guide:** [DESIGN_SYSTEM_SHOWCASE.md](./DESIGN_SYSTEM_SHOWCASE.md#logo-collection)

**Quick Example:**
```tsx
import { ApexLogo } from '@/components/icons/ApexLogo';
import { LuxurySerif } from '@/components/brandmarks/LuxurySerif';

<div className="flex items-center gap-4">
  <ApexLogo size={48} color="#d4af37" />
  <LuxurySerif text="LII" color="#d4af37" size="lg" />
</div>
```

---

## Filters & Animations

### CSS Filters Available

#### Color Filters
```css
/* Gold Tint */
filter: sepia(100%) saturate(300%) hue-rotate(30deg);

/* Monochrome */
filter: grayscale(100%);

/* High Contrast */
filter: contrast(150%);

/* Brightness Adjust */
filter: brightness(110%);

/* Blur Effect */
filter: blur(8px);
```

**Usage in Tailwind:**
```tsx
<img
  src="/product.jpg"
  className="hover:filter hover:sepia hover:saturate-150 transition-all duration-300"
  alt="Product"
/>
```

### Animation Library

#### Tailwind Animate (Built-in)

**Available Animations:**
```tsx
// Fade In
<div className="animate-in fade-in duration-500">Content</div>

// Slide In from Bottom
<div className="animate-in slide-in-from-bottom duration-500">Content</div>

// Slide In from Top
<div className="animate-in slide-in-from-top duration-500">Content</div>

// Zoom In
<div className="animate-in zoom-in duration-500">Content</div>

// Spin
<div className="animate-spin">Loading...</div>

// Ping
<div className="animate-ping">Notification</div>

// Bounce
<div className="animate-bounce">Scroll</div>
```

#### Custom Animations

**Hover Scale:**
```tsx
<div className="transition-transform duration-300 hover:scale-105">
  Scale on hover
</div>
```

**Hover Lift:**
```tsx
<div className="transition-all duration-300 hover:-translate-y-2 hover:shadow-xl">
  Lift on hover
</div>
```

**Floating Effect:**
```tsx
<div className="animate-bounce">
  Floating element
</div>
```

**Gradient Animation:**
```css
@keyframes gradient-shift {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

.gradient-animate {
  background: linear-gradient(270deg, #d4af37, #c19d2e, #e0c050);
  background-size: 600% 600%;
  animation: gradient-shift 8s ease infinite;
}
```

**Usage:**
```tsx
<div className="gradient-animate">Animated gradient</div>
```

#### Canvas Animations

**Examples in codebase:**
1. **PodcastShowcase**: Floating orbs animation
2. **Hero**: Particle effects (if implemented)
3. **Background**: Ambient animations

**Creating Custom Canvas Animation:**
```tsx
useEffect(() => {
  const canvas = canvasRef.current;
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Animation logic here
  const animate = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Draw frame
    requestAnimationFrame(animate);
  };

  animate();
}, []);
```

---

## Tailwind Components Review

### Radix UI Components (48 total)

#### ✅ Essential Components (Keep)
- **Button** - Critical for CTAs
- **Card** - Product cards, content containers
- **Dialog** - Modals, confirmations
- **Dropdown Menu** - Navigation menus
- **Form** - All form handling
- **Input** - Text inputs
- **Select** - Dropdowns
- **Tabs** - Tabbed interfaces
- **Toast** - Notifications
- **Tooltip** - Help text
- **Badge** - Status indicators
- **Skeleton** - Loading states
- **Alert** - Warnings, messages

#### ⚠️ Conditional Components (Review Usage)
- **Accordion** - If used for FAQs or product specs
- **Carousel** - If used for image galleries
- **Pagination** - If product listing pagination exists
- **Progress** - If progress bars used
- **Sidebar** - If admin panel exists
- **Table** - If data tables used

#### ❌ Consider Removing (If Unused)
- **Calendar** - Only if no date pickers
- **Chart** - Only if no data visualization
- **Command** - Only if no command palette
- **Input OTP** - Only if no 2FA
- **Menubar** - Only if not used
- **Navigation Menu** - If using custom nav
- **Resizable** - If no resizable panels
- **Toggle Group** - If not used

### Bundle Size Optimization

**Current Bundle Sizes:**
- React Vendor: 272.27 KB
- Main Vendor: 122.37 KB
- CSS: 115.35 KB

**Optimization Steps:**

1. **Analyze unused components:**
```bash
npx vite-bundle-visualizer
```

2. **Remove unused Radix components:**
```bash
# Remove unused component
rm src/components/ui/calendar.tsx

# Update imports throughout codebase
```

3. **Lazy load heavy components:**
```tsx
const ProductDetail = lazy(() => import('./pages/ProductDetail'));
const Checkout = lazy(() => import('./pages/Checkout'));
```

4. **Purge unused Tailwind classes:**
```js
// tailwind.config.ts
export default {
  content: ['./src/**/*.{ts,tsx}'], // Ensure correct paths
  // ...
}
```

**Expected Impact:**
- Remove 5-10 unused components: -15-30 KB
- Lazy loading: -20-40 KB initial bundle
- Tailwind purge: -10-20 KB CSS

**Target Bundle Sizes:**
- React Vendor: <250 KB
- Main Vendor: <100 KB
- CSS: <100 KB

---

## Template Best Practices

### Design Principles

#### 1. Mobile-First Design
```tsx
// Good: Mobile-first, desktop enhances
<div className="text-base md:text-lg lg:text-xl">

// Bad: Desktop-first
<div className="text-xl md:text-base">
```

#### 2. Consistent Spacing
```tsx
// Good: Use spacing scale (4, 6, 8, 12, 16, 20, 24)
<section className="py-20">
  <div className="container px-6 mb-12">

// Bad: Random spacing
<section className="py-17">
  <div className="container px-7 mb-11">
```

#### 3. Semantic HTML
```tsx
// Good: Semantic elements
<section>
  <h2>Heading</h2>
  <article>Content</article>
</section>

// Bad: Div soup
<div>
  <div>Heading</div>
  <div>Content</div>
</div>
```

#### 4. Accessibility
```tsx
// Good: Accessible button
<button
  aria-label="Add to cart"
  className="focus:ring-2 focus:ring-lii-gold"
>
  <ShoppingCart className="h-5 w-5" />
</button>

// Bad: Non-accessible
<div onClick={handleClick}>
  <ShoppingCart />
</div>
```

### Performance Tips

#### 1. Image Optimization
```tsx
// Good: Responsive images
<img
  src="/product.jpg"
  srcSet="/product-sm.jpg 640w, /product-md.jpg 1024w, /product-lg.jpg 1920w"
  sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
  alt="Product name"
  loading="lazy"
/>

// Bad: Large unoptimized image
<img src="/product-full.jpg" />
```

#### 2. Code Splitting
```tsx
// Good: Lazy load routes
const Shop = lazy(() => import('./pages/Shop'));
const Checkout = lazy(() => import('./pages/Checkout'));

<Suspense fallback={<LoadingSpinner />}>
  <Routes>
    <Route path="/shop" element={<Shop />} />
    <Route path="/checkout" element={<Checkout />} />
  </Routes>
</Suspense>
```

#### 3. Memoization
```tsx
// Good: Memoize expensive calculations
const filteredProducts = useMemo(() => {
  return products.filter(p => p.category === selectedCategory);
}, [products, selectedCategory]);

// Bad: Recalculate on every render
const filteredProducts = products.filter(p => p.category === selectedCategory);
```

### Component Organization

**Recommended Structure:**
```
src/
├── components/
│   ├── ui/           # Radix UI primitives
│   ├── product/      # Product-specific components
│   ├── checkout/     # Checkout flow components
│   ├── logo/         # Logo components
│   ├── brandmarks/   # Brandmark components
│   └── icons/        # Icon components
├── pages/            # Route pages
├── hooks/            # Custom hooks
├── utils/            # Utilities
└── types/            # TypeScript types
```

---

## FAQ

### General Questions

**Q: Does the YouTube channel integration work?**
A: Yes, the PodcastShowcase component is fully functional with YouTube icon integration. You need to add your YouTube video ID and channel URL to embed videos. See [YouTube & Media Integration](#youtube--media-integration).

**Q: Where are the apparel examples?**
A: Apparel mockups are in `src/assets/products/` and displayed on:
- Homepage Product Showcase
- Shop page at `/shop`
- Individual product details at `/product/:id`

See [Apparel & Products](#apparel--products) for details.

**Q: How many logos are available?**
A: **39 total logos:**
- 4 main brand logos
- 16 icon logos
- 15 brandmarks
- 4 seasonal bird logos

See [Logo & Branding Assets](#logo--branding-assets).

**Q: What UI components are included?**
A: **48 Radix UI components** including buttons, cards, forms, dialogs, tooltips, and more. See [UI Components](./DESIGN_SYSTEM_SHOWCASE.md#ui-components).

### Technical Questions

**Q: How do I create a new page template?**
A: Follow the [Template Customization](#template-customization) guide.

**Q: Which Tailwind components should I remove?**
A: See [Tailwind Components Review](#tailwind-components-review) for recommendations.

**Q: How do I add custom animations?**
A: See [Filters & Animations](#filters--animations) for examples.

**Q: What's the bundle size target?**
A: Target <450 KB total (React <250 KB, Main <100 KB, CSS <100 KB).

**Q: How do I optimize images?**
A: Use responsive images with `srcSet`, lazy loading, and WebP format.

### Design Questions

**Q: What fonts does the brand use?**
A:
- **Display**: Playfair Display (headings)
- **UI**: Inter Variable (body)
- **Mono**: JetBrains Mono (code)

**Q: What are the brand colors?**
A:
- **Primary**: Championship Gold (#d4af37)
- **Background**: Carbon Black (#0B0B0C)
- **Text**: Cloud (#E6E9EF), Ash (#8C93A3)

See [Color System](./DESIGN_SYSTEM_SHOWCASE.md#color-system).

**Q: What's the recommended spacing scale?**
A: Use multiples of 4: 4px, 8px, 12px, 16px, 24px, 32px, 48px, 64px, 96px.

**Q: How do I maintain accessibility?**
A: Follow [Accessibility Standards](./DESIGN_SYSTEM_SHOWCASE.md#accessibility-standards):
- 4.5:1 contrast ratio for text
- Keyboard navigation
- ARIA labels
- Semantic HTML

### Business Questions

**Q: Can I add more products?**
A: Yes, add product data to the database and create mockup SVGs in `src/assets/products/`.

**Q: How do I change pricing?**
A: Update product data in your Supabase database or local product definitions.

**Q: How do I integrate with my YouTube channel?**
A: Add your video ID and channel URL in PodcastShowcase. See [YouTube & Media Integration](#youtube--media-integration).

**Q: How do I customize the brand?**
A: Update design tokens in `src/theme/tokens.ts` and replace logos in `src/components/logo/` and `src/components/icons/`.

---

## Additional Resources

### Documentation
- [Project Reference](./PROJECT_REFERENCE.md) - Health scores, metrics, acronyms
- [Design System Showcase](./DESIGN_SYSTEM_SHOWCASE.md) - Complete design system
- [API Documentation](./API.md) - API endpoints
- [Architecture Guide](./ARCHITECTURE.md) - System architecture

### External Links
- [React Documentation](https://react.dev)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Radix UI](https://radix-ui.com)
- [Vite](https://vitejs.dev)
- [Supabase](https://supabase.com/docs)

---

**Maintained by:** Development Team
**Last Review:** 2025-11-12
**Next Review:** 2026-01-12

**Questions?** Create an issue using the [issue template](../.github/ISSUE_TEMPLATE.md).
