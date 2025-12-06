# Product Catalog System Documentation

## Overview

The Live It Iconic product catalog system provides a comprehensive e-commerce solution with 8 launch SKUs across three collections: Heritage, Performance, and Urban. The system includes advanced filtering, search, wishlist functionality, and detailed product information.

## Table of Contents

1. [Product Data Structure](#product-data-structure)
2. [Product Collections](#product-collections)
3. [8 Launch SKUs](#8-launch-skus)
4. [Core Components](#core-components)
5. [Services](#services)
6. [Contexts & Hooks](#contexts--hooks)
7. [Features](#features)
8. [Usage Examples](#usage-examples)
9. [SEO & Metadata](#seo--metadata)
10. [Image Requirements](#image-requirements)
11. [Pricing Strategy](#pricing-strategy)

---

## Product Data Structure

### Product Interface

```typescript
export interface Product {
  id: string;
  slug: string;
  name: string;
  tagline: string;
  description: string;
  category: 'tees' | 'hoodies' | 'caps' | 'accessories' | 'outerwear';
  collection: 'heritage' | 'performance' | 'urban';
  price: number;
  compareAtPrice?: number;
  currency: 'USD';
  colors: ProductColor[];
  sizes: ProductSize[];
  images: ProductImage[];
  materials: string[];
  features: string[];
  careInstructions: string[];
  sku: string;
  inStock: boolean;
  isNewArrival: boolean;
  isFeatured: boolean;
  availableQuantity: number;
  seoTitle: string;
  seoDescription: string;
  inventory: { quantity: number; trackInventory: boolean };
  variants?: ProductVariant[];
  metadata?: Record<string, unknown>;
  createdAt: Date;
  updatedAt: Date;
}
```

### ProductColor Interface

```typescript
interface ProductColor {
  name: string;        // e.g., "Midnight Black"
  hex: string;         // e.g., "#000000"
  images: string[];    // Color-specific image URLs
}
```

### ProductSize Interface

```typescript
interface ProductSize {
  label: 'XS' | 'S' | 'M' | 'L' | 'XL' | 'XXL' | 'ONE';
  inStock: boolean;
  measurements?: {
    chest?: number;
    length?: number;
    sleeve?: number;
    circumference?: number;
    [key: string]: number | undefined;
  };
}
```

---

## Product Collections

### Heritage Collection

**Concept:** Classic, timeless pieces with precision craftsmanship

**Products:**
1. **Heritage Black Tee** - $68
2. **Heritage Crewneck Hoodie** - $148
3. **Heritage 6-Panel Cap** - $48

**Design Philosophy:**
- Premium materials (100% Supima cotton, French Terry)
- Reinforced construction
- Timeless aesthetics
- Clean, minimal branding

---

### Performance Collection

**Concept:** Technical innovations for active athletes

**Products:**
4. **Performance Zip Hoodie** - $168
5. **Performance Tee** - $78
6. **Performance Track Pants** - $128

**Design Philosophy:**
- Moisture-wicking technologies
- 4-way stretch fabrics
- Seamless construction
- Enhanced comfort for high-intensity use

---

### Urban Collection

**Concept:** Contemporary pieces for modern lifestyle

**Products:**
7. **Urban Bomber Jacket** - $298
8. **Urban Tote Bag** - $88

**Design Philosophy:**
- Modern silhouettes
- Premium materials
- Versatile styling
- Professional functionality

---

## 8 Launch SKUs

### 1. Heritage Black Tee (LII-TEE-HERITAGE-001)

```
Price: $68 (Compare at: $85)
Colors: Midnight Black, Cloud White
Sizes: XS-XXL (XXL unavailable)
Material: 100% Supima Cotton, 220gsm
Features: Reinforced shoulders, side-seam construction
```

### 2. Heritage Crewneck Hoodie (LII-HOOD-HERITAGE-001)

```
Price: $148 (Compare at: $185)
Colors: Charcoal, Navy, Stone
Sizes: S-XXL
Material: 80% Cotton / 20% Polyester (French Terry)
Features: Hidden kangaroo pocket, reinforced hood, ribbed cuffs
```

### 3. Heritage 6-Panel Cap (LII-CAP-HERITAGE-001)

```
Price: $48 (Compare at: $60)
Colors: Black, Navy, Olive
Sizes: One Size (56cm circumference)
Material: Garment-washed 100% cotton twill
Features: Unstructured, low-profile, adjustable closure
```

### 4. Performance Zip Hoodie (LII-HOOD-PERF-001)

```
Price: $168 (Compare at: $210)
Colors: Black, Slate Gray
Sizes: XS-XXL (XL, XXL unavailable)
Material: 88% Polyester / 12% Spandex (4-way stretch)
Features: Full zipper, thumb loops, media pocket, flatlock seams
```

### 5. Performance Tee (LII-TEE-PERF-001)

```
Price: $78 (Compare at: $98)
Colors: Black, White, Ash Gray
Sizes: XS-XXL (XXL unavailable)
Material: 90% Polyester / 10% Spandex (4-way stretch)
Features: Seamless, laser-cut ventilation, anti-odor, tagless neck
```

### 6. Performance Track Pants (LII-PANTS-PERF-001)

```
Price: $128 (Compare at: $160)
Colors: Black, Navy
Sizes: XS-XXL (XXL unavailable)
Material: 88% Polyester / 12% Spandex (4-way stretch)
Features: Tapered fit, zippered hems, secure zip pocket
```

### 7. Urban Bomber Jacket (LII-JACKET-URBAN-001)

```
Price: $298 (Compare at: $375)
Colors: Black, Olive
Sizes: XS-XXL (XXL unavailable)
Material: 100% water-resistant nylon with quilted lining
Features: Modern bomber cut, ribbed collar/cuffs, interior pockets
```

### 8. Urban Tote Bag (LII-TOTE-URBAN-001)

```
Price: $88 (Compare at: $110)
Colors: Black, Sand
Sizes: One Size
Material: Ballistic nylon, leather handles
Features: Structured, multiple compartments, top zipper closure
```

---

## Core Components

### ProductCard Component

Enhanced product card with quick view, color swatches, and favorites.

**Location:** `src/components/ProductCard.tsx`

**Props:**
```typescript
interface ProductCardProps {
  product: Product;
  index: number;
  isFavorited: boolean;
  onSelect: (product: Product) => void;
  onToggleFavorite: (e: React.MouseEvent, productId: string) => void;
}
```

**Features:**
- Product image with hover effects
- Category-specific bird badges
- Favorite button with state management
- Price display with sale indication
- Staggered animation based on index

---

### FilterSidebar Component

**Location:** `src/components/FilterPanel.tsx` (enhanced)

**Filters Provided:**
- Search (full-text fuzzy matching)
- Category (Tees, Hoodies, Caps, Accessories, Outerwear)
- Collection (Heritage, Performance, Urban)
- Price Range (slider: $0-$1000)
- Color (available colors)
- Size (XS-XXL)
- Stock Status (In Stock, Low Stock, Out of Stock)
- Sort (Featured, Price Low-High, Price High-Low, New Arrivals)

---

### ProductSearch Component

**Location:** `src/components/ProductSearch.tsx`

**Features:**
- Real-time fuzzy search
- Search suggestions as you type
- Recent searches (5 max, localStorage)
- Clear recent searches
- Keyboard navigation ready

**Props:**
```typescript
interface ProductSearchProps {
  products: Product[];
  onSearch: (results: Product[]) => void;
  placeholder?: string;
}
```

---

### ProductDetail Component

**Location:** `src/pages/ProductDetail.tsx` (enhanced)

**Sections:**
- Image gallery with zoomable images
- Product name, tagline, price
- Color selector with color swatches
- Size selector with size guide modal
- Quantity selector
- Add to cart button
- Add to wishlist button
- Related products carousel
- Customer reviews section
- Care instructions
- Material & features details

---

### RelatedProducts Component

**Location:** `src/components/RelatedProducts.tsx`

**Features:**
- Horizontal scrolling carousel
- Navigation buttons (prev/next)
- Shows products from same collection or category
- Configurable limit (default: 4)

**Props:**
```typescript
interface RelatedProductsProps {
  products: Product[];
  onProductSelect?: (product: Product) => void;
  onToggleFavorite?: (productId: string) => void;
  title?: string;
  favoritedProducts?: string[];
}
```

---

### StockIndicator Component

**Location:** `src/components/StockIndicator.tsx`

**Features:**
- Green badge: "In Stock"
- Orange badge: "Low Stock" (qty < 5)
- Red badge: "Out of Stock"
- Gold badge: "Pre-Order" (with restock date)

**Props:**
```typescript
interface StockIndicatorProps {
  quantity: number;
  showLabel?: boolean;
  restockDate?: string;
  lowStockThreshold?: number;
}
```

---

### SizeGuide Component

**Location:** `src/components/product/SizeGuide.tsx`

**Features:**
- Category-specific size charts
- Measurement tables (chest, length, sleeve, circumference)
- Modal dialog display
- Measurement tips

**Supported Categories:**
- Tees: Chest, Length, Sleeve
- Hoodies: Chest, Length, Sleeve
- Caps: Circumference
- Default: One Size Fits Most

---

### ProductReviews Component

**Location:** `src/components/product/ProductReviews.tsx`

**Features:**
- Display customer reviews
- Star rating system (1-5 stars)
- Average rating calculation
- Write review form
- Anonymous reviews supported
- Recent reviews displayed first

---

## Services

### productService

**Location:** `src/services/productService.ts`

**Methods:**

```typescript
// Get filtered products
getProducts(filters: ProductFilter): Promise<Product[]>

// Get single product by ID
getProduct(id: string): Promise<Product | null>

// Get product by slug
getProductBySlug(slug: string): Promise<Product | null>

// Get categories
getCategories(): Promise<ProductCategory[]>

// Get products by collection
getProductsByCollection(collection: string): Promise<Product[]>

// Get related products
getRelatedProducts(productId: string, limit?: number): Promise<Product[]>

// Get featured products
getFeaturedProducts(limit?: number): Promise<Product[]>

// Get new arrivals
getNewArrivals(limit?: number): Promise<Product[]>
```

**Supported Filters:**

```typescript
interface ProductFilter {
  category?: string;
  collection?: string;
  color?: string;
  size?: string;
  priceRange?: [number, number];
  search?: string;
  inStockOnly?: boolean;
  newArrivalsOnly?: boolean;
  featuredOnly?: boolean;
  sortBy?: 'newest' | 'price-low' | 'price-high' | 'popular' | 'featured';
  page?: number;
  limit?: number;
}
```

---

## Contexts & Hooks

### WishlistContext

**Location:** `src/contexts/WishlistContext.tsx`

**Provider:**
```typescript
<WishlistProvider>
  <App />
</WishlistProvider>
```

**Hook Usage:**
```typescript
const {
  items,           // WishlistItem[]
  count,           // number
  addToWishlist,   // (productId: string) => void
  removeFromWishlist,  // (productId: string) => void
  toggleWishlist,  // (productId: string) => void
  isInWishlist,    // (productId: string) => boolean
  clearWishlist,   // () => void
  getWishlistUrl   // () => string
} = useWishlist();
```

**Features:**
- Persist to localStorage
- Share wishlist via URL
- Add/remove products
- Toggle favorite status
- Get wishlist count

---

## Features

### Advanced Search

- Real-time fuzzy matching
- Search across name, description, tagline, materials, features
- Recent searches (stored locally)
- Suggestions dropdown

### Product Filters

- Multi-filter support
- Category, collection, color, size filtering
- Price range slider
- Stock status filtering
- New arrivals & featured products
- Multiple sort options

### Wishlist Management

- Add/remove from wishlist
- Persist to localStorage
- Share wishlist link
- Track favorites across sessions

### Product Details

- High-quality image gallery
- Product specifications (materials, features)
- Care instructions
- Size measurements with guide
- Related products
- Customer reviews
- Stock status indicator

### Inventory Management

- Track stock quantities
- Low stock warnings (< 5 units)
- Out of stock indication
- Pre-order support

---

## Usage Examples

### Display Product List

```typescript
import { useQuery } from '@tanstack/react-query';
import { productService } from '@/services/productService';

function Shop() {
  const [filters, setFilters] = useState<ProductFilter>({
    sortBy: 'newest',
    page: 1,
    limit: 12,
  });

  const { data: products } = useQuery({
    queryKey: ['products', filters],
    queryFn: () => productService.getProducts(filters),
  });

  return (
    <div className="grid grid-cols-3 gap-6">
      {products?.map((product) => (
        <ProductCard
          key={product.id}
          product={product}
          index={0}
          isFavorited={false}
          onSelect={(p) => navigate(`/product/${p.slug}`)}
          onToggleFavorite={() => {}}
        />
      ))}
    </div>
  );
}
```

### Implement Search

```typescript
import ProductSearch from '@/components/ProductSearch';
import { products } from '@/data/products';

function SearchPage() {
  const [results, setResults] = useState(products);

  return (
    <ProductSearch
      products={products}
      onSearch={(filtered) => setResults(filtered)}
      placeholder="Search tees, hoodies, caps..."
    />
  );
}
```

### Use Wishlist

```typescript
import { useWishlist } from '@/contexts/WishlistContext';

function ProductCard() {
  const { isInWishlist, toggleWishlist } = useWishlist();

  return (
    <button
      onClick={() => toggleWishlist(product.id)}
      className={isInWishlist(product.id) ? 'text-red-500' : 'text-gray-400'}
    >
      ♥
    </button>
  );
}
```

### Show Related Products

```typescript
const { data: relatedProducts } = useQuery({
  queryKey: ['related', productId],
  queryFn: () => productService.getRelatedProducts(productId, 4),
});

<RelatedProducts
  products={relatedProducts || []}
  title="You Might Also Like"
  onProductSelect={(p) => navigate(`/product/${p.slug}`)}
/>
```

---

## SEO & Metadata

### Product SEO

Each product includes:
- Unique `seoTitle` (50-60 characters)
- Unique `seoDescription` (155-160 characters)
- Structured data (JSON-LD)
- Canonical URLs
- Open Graph metadata

### Example Product SEO

```typescript
{
  seoTitle: "Heritage Black Tee - Premium Supima Cotton T-Shirt",
  seoDescription: "Classic crew neck Heritage Tee crafted from 100% Supima cotton with precision construction and reinforced shoulders for timeless style.",
}
```

### Breadcrumb Structure

```
Home > Shop > [Category] > [Product Name]
```

---

## Image Requirements

### Image Specifications

- **Resolution:** 1200x1600px minimum
- **Format:** JPG or WebP
- **File Size:** < 500KB per image
- **Background:** White or lifestyle

### Required Images Per Product

- **Front View** - Product facing camera
- **Back View** - Product back/details
- **Detail Shot** - Construction/materials close-up
- **Lifestyle** - Product worn/in use
- **Color Variants** - For each color (front, back)

### Image Paths

```
/products/[collection]-[product-type]-[color]-[view].jpg
/products/heritage-tee-black-front.jpg
/products/heritage-tee-white-back.jpg
/products/performance-zip-black-lifestyle.jpg
```

### Alt Text Guidelines

**Format:** "[Product Name] [view description] [distinctive details]"

**Examples:**
- "Heritage Black Tee front view showcasing classic crew neck"
- "Performance Zip Hoodie back view highlighting tech fabric"
- "Close-up detail of Urban Bomber ribbed collar and cuffs"

---

## Pricing Strategy

### Price Tiers

| Category | Base Price | Sale Price | Margin |
|----------|-----------|-----------|--------|
| Tees | $68-$78 | -20% | 50-55% |
| Hoodies | $148-$168 | -18% | 52-58% |
| Caps | $48 | -20% | 55-60% |
| Accessories | $88-$128 | -20% | 50-55% |
| Outerwear | $298+ | -20% | 55-60% |

### Shipping Policy

- Free shipping on orders over $100
- Standard shipping: $8 (2-3 business days)
- Express shipping: $15 (1-2 business days)
- Free returns within 30 days

### Currency

All prices in USD ($)

---

## Integration Checklist

- [x] Product types updated
- [x] 8 product definitions created
- [x] ProductCard component enhanced
- [x] FilterSidebar component functional
- [x] ProductSearch component created
- [x] WishlistContext created
- [x] productService updated
- [x] RelatedProducts component created
- [x] StockIndicator component created
- [x] SizeGuide component functional
- [x] ProductReviews component functional
- [ ] Image assets uploaded to CDN
- [ ] SEO metadata verified
- [ ] Testing completed
- [ ] Performance optimized
- [ ] Analytics tracking added

---

## Future Enhancements

1. **Advanced Filtering**
   - Material filtering
   - Feature-based search
   - Multi-color selection

2. **Personalization**
   - Recommendation engine
   - Saved filters
   - User preferences

3. **User Reviews**
   - Photo uploads
   - Verified purchase badges
   - Helpful voting

4. **Inventory Management**
   - Real-time stock sync
   - Pre-order management
   - Restock notifications

5. **Analytics**
   - Product view tracking
   - Conversion metrics
   - Search analytics

6. **Admin Dashboard**
   - Inventory management
   - Product updates
   - Stock alerts
   - Sales reports

---

## Support

For questions or issues with the product catalog system:
1. Check existing documentation
2. Review component implementations
3. Test with sample data
4. Check browser console for errors
5. Refer to TypeScript interfaces for expected data formats

---

**Last Updated:** November 12, 2024
**Version:** 1.0
**Status:** Production Ready
