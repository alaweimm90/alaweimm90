# Performance Optimization Guide

## Core Web Vitals Targets

Live It Iconic is optimized to meet Google's Core Web Vitals standards:

| Metric | Target | Status |
|--------|--------|--------|
| **LCP** (Largest Contentful Paint) | ≤ 2.5s | ✓ Optimized |
| **INP** (Interaction to Next Paint) | < 200ms | ✓ Optimized |
| **CLS** (Cumulative Layout Shift) | < 0.1 | ✓ Optimized |
| **Performance Score** | ≥ 90 | ✓ Targeted |
| **Accessibility Score** | ≥ 95 | ✓ Targeted |

## Performance Optimizations Implemented

### 1. Image Optimization

#### OptimizedImage Component
Location: `src/components/OptimizedImage.tsx`

The `OptimizedImage` component provides:
- **Modern formats**: AVIF (best) → WebP (good) → JPEG (fallback)
- **Lazy loading**: IntersectionObserver defers below-fold images
- **Blur placeholders**: Smooth fade-in with placeholder image
- **Explicit dimensions**: Prevents Cumulative Layout Shift (CLS)
- **Responsive sizing**: srcset for different viewport sizes
- **Priority loading**: `priority={true}` for LCP images (hero, above-fold)

#### Usage Example:
```tsx
import OptimizedImage from '@/components/OptimizedImage';

<OptimizedImage
  src={jpegImage}
  srcWebp={webpImage}
  srcAvif={avifImage}
  alt="Hero image with athlete and supercar"
  width={1200}
  height={800}
  priority={true} // Load immediately for LCP images
  placeholder={blurImageData}
  sizes="(max-width: 640px) 100vw, (max-width: 1024px) 80vw, 1200px"
/>
```

#### Image Format Conversion
Convert existing JPEGs to AVIF/WebP:

```bash
# Install tools
brew install libavif libwebp

# Convert to AVIF (best compression)
avifenc input.jpg output.avif --quality 75 --speed 0

# Convert to WebP (good compression)
cwebp input.jpg -o output.webp -quality 75

# Batch conversion (macOS)
for f in *.jpg; do
  avifenc "$f" "${f%.jpg}.avif" --quality 75 --speed 0
  cwebp "$f" -o "${f%.jpg}.webp" -quality 75
done
```

### 2. Font Loading Optimization

#### Configuration
Location: `index.html`

Optimizations:
- **font-display: swap**: Shows fallback text immediately, swaps custom font when loaded
- **Preload critical fonts**: Preload Playfair Display (heading font) for LCP
- **Variable fonts**: Inter Variable font (single file vs multiple weights)
- **Google Fonts API**: Uses `?display=swap` parameter for optimal loading

#### Best Practices:
- Only preload fonts actually needed above-the-fold
- Use variable fonts to reduce number of font files
- Use system fonts as fallbacks to prevent FOIT (Flash of Invisible Text)

### 3. Code Splitting & Bundling

#### Configuration
Location: `vite.config.ts`

Chunk Strategy:
```
react-core/         → React + DOM (stable, cached long-term)
react-router/       → Router (stable)
radix-ui/           → Radix UI components
icons/              → Lucide React icons
style-utils/        → TailwindCSS utilities
api-vendor/         → TanStack Query + Supabase
forms/              → Form libraries (React Hook Form, Zod)
cart/               → Cart context & state
cart-ui/            → Cart components
checkout/           → Checkout page
admin/              → Admin features
shop/               → Shop/collection pages
vendor/             → Other dependencies
main/               → Application code
```

Benefits:
- **Long-term caching**: Vendor chunks rarely change, cached for 1 year
- **Parallel loading**: Smaller chunks load faster in parallel
- **Lazy route loading**: Pages load on-demand, not in initial bundle
- **Tree-shaking**: Unused code removed during build

### 4. React Performance Optimizations

#### Hooks & Utilities
Location: `src/hooks/usePerformanceOptimization.ts`

Available hooks:
```tsx
// Debounce expensive operations (search, resize)
const debouncedSearch = useDebounce(handleSearch, 300);

// Throttle frequent events (scroll, mousemove)
const throttledScroll = useThrottle(handleScroll, 100);

// Measure component render time
useRenderTime('ProductGrid');

// Run work when browser is idle
useIdleCallback(() => prefetchData());

// Track intersection for lazy loading
const isVisible = useIntersectionObserver(ref);

// Measure interaction latency (for INP)
const measureEnd = measureInteraction('click-to-paint');
handleClick();
measureEnd();
```

#### Component Optimization Pattern:
```tsx
import { useCallback, useMemo } from 'react';

const ProductCard = React.memo(({ product, onSelect }) => {
  // Use useCallback for props passed to children
  const handleClick = useCallback(() => {
    onSelect(product.id);
  }, [product.id, onSelect]);

  // Use useMemo for expensive computations
  const discountedPrice = useMemo(() => {
    return calculateDiscount(product.price, product.discount);
  }, [product.price, product.discount]);

  return (
    <div onClick={handleClick}>
      <h3>{product.name}</h3>
      <p>${discountedPrice}</p>
    </div>
  );
});

ProductCard.displayName = 'ProductCard';
```

### 5. API Optimization

#### Request Caching
Location: `src/lib/api-cache.ts`

Features:
- **Deduplication**: Multiple identical requests reuse the same response
- **Stale-while-revalidate**: Return cached data immediately, update in background
- **TTL-based expiration**: Configurable cache lifetime
- **Memory management**: Automatic cleanup prevents memory leaks

#### Usage Example:
```tsx
import { apiCache, staleWhileRevalidate } from '@/lib/api-cache';

// Basic fetch with caching (5 minute TTL)
const products = await apiCache.fetch('/api/products', {
  cache: { ttl: 5 * 60 * 1000 }
});

// Stale-while-revalidate pattern
const data = await staleWhileRevalidate('/api/user', {
  ttl: 10 * 60 * 1000,
  onUpdate: (fresh) => updateUI(fresh)
});

// Prefetch critical data
import { prefetchCriticalData } from '@/lib/api-cache';

prefetchCriticalData([
  { url: '/api/products', priority: 'high', ttl: 5 * 60 * 1000 },
  { url: '/api/categories', priority: 'high', ttl: 10 * 60 * 1000 },
  { url: '/api/user', priority: 'low', ttl: 5 * 60 * 1000 }
]);
```

### 6. Web Vitals Monitoring

#### Setup
Location: `src/lib/webVitals.ts`

Automatically tracks:
- **FCP** (First Contentful Paint): ~1.8s target
- **LCP** (Largest Contentful Paint): ≤2.5s target
- **INP** (Interaction to Next Paint): <200ms target
- **CLS** (Cumulative Layout Shift): <0.1 target
- **TTFB** (Time to First Byte): ~600ms target

Initialized in `src/main.tsx`:
```tsx
import { initWebVitals } from '@/lib/webVitals';

if (import.meta.env.PROD) {
  initWebVitals(); // Start tracking metrics
}
```

Metrics are automatically sent to Google Analytics if gtag is available.

#### Getting Current Metrics:
```tsx
import { getCoreWebVitals } from '@/lib/webVitals';

const metrics = getCoreWebVitals();
console.log('LCP:', metrics.LCP, 'ms');
console.log('FCP:', metrics.FCP, 'ms');
```

### 7. Asset Caching Strategy

#### Headers Configuration
Location: `public/_headers`

Caching rules:
- **Hashed assets** (`/assets/*`): 1 year cache (immutable)
- **JavaScript/CSS**: 1 year cache (immutable)
- **Fonts**: 1 year cache (immutable)
- **Images**: 1 year cache (immutable)
- **Service worker** (`/sw.js`): No cache (always fresh)
- **HTML files**: No cache (always revalidate)
- **API responses**: No cache (always fetch)

Browsers use hash in filename to validate freshness:
```html
<!-- Old version (cached) -->
<script src="/assets/main-a1b2c3d4.js"></script>

<!-- New version (always fetched if filename changes) -->
<script src="/assets/main-e5f6g7h8.js"></script>
```

### 8. Bundle Analysis

View bundle composition after build:
```bash
npm run build
# Open dist/stats.html in browser
```

The visualizer shows:
- Chunk sizes (gzipped and uncompressed)
- Module breakdown
- Opportunities for optimization

### 9. Lighthouse CI

#### Configuration
Location: `.lighthouserc.js`

Runs automated performance testing:
```bash
# Manual audit
npm run build && npm run preview
# Then run: lighthouse http://localhost:4173/

# Or with Lighthouse CI
npm install -g @lhci/cli@next
lhci autorun
```

Thresholds:
- Performance: ≥90
- Accessibility: ≥95
- Best Practices: ≥95
- SEO: 100

### 10. Build Optimizations

#### Minification
- **JavaScript**: Terser (removes console, debugger, unused code)
- **CSS**: TailwindCSS purging + CSS minification
- **Assets**: Hash-based naming for cache busting

#### Development vs Production
```tsx
// Remove in production build
if (import.meta.env.DEV) {
  console.log('Debug info');
}

// Only in production
if (import.meta.env.PROD) {
  initWebVitals();
  registerServiceWorker();
}
```

## Testing & Validation

### 1. Local Performance Testing

```bash
# Run development server
npm run dev

# In another terminal, simulate slow network
# Open DevTools → Network → Throttling (Slow 3G)

# Reload and observe metrics in console
```

### 2. Lighthouse Audit

```bash
# Install Lighthouse globally
npm install -g lighthouse

# Run audit
lighthouse https://liveiticonic.com --view

# Or against local build
npm run build && npm run preview
lighthouse http://localhost:4173/ --view
```

### 3. Chrome DevTools

Open DevTools → Performance tab:
1. Click Record
2. Interact with the page
3. Click Stop
4. Analyze the timeline:
   - Look for long tasks (red segments)
   - Check for CLS indicators
   - Review FCP/LCP markers

### 4. WebPageTest

Visit [WebPageTest.org](https://www.webpagetest.org/):
1. Enter site URL
2. Select test location & browser
3. Run test
4. Review detailed metrics and waterfall

## Performance Checklist

- [ ] Images use OptimizedImage component
- [ ] Hero/above-fold images have `priority={true}`
- [ ] Below-fold images use lazy loading
- [ ] All images have explicit width/height (prevent CLS)
- [ ] Fonts preloaded in index.html
- [ ] font-display: swap used for custom fonts
- [ ] API calls use caching (apiCache)
- [ ] List components use React.memo
- [ ] Expensive computations use useMemo
- [ ] Event handlers use useCallback
- [ ] Scroll handlers use throttle
- [ ] Search/resize handlers use debounce
- [ ] Pages are lazy loaded via route splitting
- [ ] Unused dependencies removed from package.json
- [ ] No console.log in production code
- [ ] Service worker registered for offline support
- [ ] Web Vitals initialized in production
- [ ] Lighthouse audit passes all targets
- [ ] Bundle analysis reviewed (dist/stats.html)
- [ ] Cache headers configured (_headers)

## Performance Budget

Target sizes for production build:
- **Main bundle**: < 250KB (gzipped)
- **React core**: < 40KB (gzipped)
- **Per-page chunk**: < 150KB (gzipped)
- **Total initial load**: < 400KB (gzipped)

Monitor with:
```bash
npm run build
# Check console output for file sizes
# Review dist/stats.html for composition
```

## Browser Support

Performance optimizations support:
- Chrome/Edge 90+
- Firefox 88+
- Safari 15+
- iOS Safari 15+
- Android Chrome 90+

Fallbacks for older browsers:
- AVIF → WebP → JPEG (for images)
- Variable fonts → Weight-specific fonts
- requestIdleCallback → setTimeout

## Links & Resources

- [Web Vitals Documentation](https://web.dev/vitals/)
- [Lighthouse Documentation](https://developers.google.com/web/tools/lighthouse)
- [Performance Best Practices](https://web.dev/performance/)
- [Image Optimization](https://web.dev/image-optimization/)
- [Font Loading Strategies](https://web.dev/font-display/)
- [Code Splitting Guide](https://webpack.js.org/guides/code-splitting/)
- [React Performance](https://react.dev/reference/react/useMemo)

## Troubleshooting

### LCP Not Improving
1. Check image size: `ls -lh src/assets/`
2. Verify image is marked with `priority={true}`
3. Check preload directives in index.html
4. Use DevTools to measure paint timing

### CLS Issues
1. Ensure all images have explicit width/height
2. Check for late-loaded content shifting layout
3. Avoid dynamic content without reserved space
4. Use `size-content-box` on containers

### INP Slow
1. Profile in DevTools Performance tab
2. Look for long tasks (>50ms)
3. Use useCallback for event handlers
4. Use useDebounce for frequent events
5. Consider moving work to Web Workers

## Continuous Monitoring

Set up performance monitoring:
1. Enable Lighthouse CI in CI/CD pipeline
2. Track metrics over time with analytics
3. Set up alerts for regressions
4. Review user experience metrics (RUM)

---

**Last Updated**: 2025-11-12
**Performance Engineer**: Live It Iconic Team
