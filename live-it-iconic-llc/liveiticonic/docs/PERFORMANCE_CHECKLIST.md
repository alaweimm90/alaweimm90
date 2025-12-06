# Performance Optimization Checklist

Complete checklist for maintaining and extending performance optimizations.

## Phase 1: Image Optimization ✓

- [x] Create OptimizedImage component
- [x] Support WebP/AVIF formats with JPEG fallback
- [x] Implement lazy loading with IntersectionObserver
- [x] Add blur placeholders for smooth fade-in
- [x] Ensure explicit width/height (prevent CLS)
- [x] Create responsive image sizes (srcset)
- [ ] Convert all product images to AVIF/WebP
- [ ] Create placeholder (blur-up) images
- [ ] Mark hero images with `priority={true}`
- [ ] Review and optimize all imported images in components

## Phase 2: Font Loading Optimization ✓

- [x] Configure font-display: swap
- [x] Preload critical fonts in index.html
- [x] Use WOFF2 format for font files
- [x] Use variable fonts (Inter Variable)
- [x] Add font fallback metrics
- [ ] Subset fonts to Latin characters only (if not using Google Fonts)
- [ ] Test font loading with slow network throttling
- [ ] Verify no FOIT (Flash of Invisible Text)

## Phase 3: Code Splitting ✓

- [x] Configure Vite manual chunks
- [x] Separate vendor bundles by function
- [x] Lazy load pages via route splitting
- [x] Add Rollup Visualizer for bundle analysis
- [x] Configure build minification (Terser)
- [x] Remove console logs from production
- [ ] Review bundle composition (npm run build)
- [ ] Check dist/stats.html for large modules
- [ ] Identify and remove unused dependencies
- [ ] Implement dynamic imports for heavy features

## Phase 4: React Performance ✓

- [x] Create performance optimization hooks
- [x] Implement useDebounce hook
- [x] Implement useThrottle hook
- [x] Create memoization utilities
- [ ] Wrap list item components with React.memo
- [ ] Add useCallback to event handler props
- [ ] Add useMemo to expensive computations
- [ ] Test component performance with DevTools
- [ ] Profile with React Profiler
- [ ] Consider virtual scrolling for large lists

## Phase 5: API Optimization ✓

- [x] Create API caching utility
- [x] Implement request deduplication
- [x] Add stale-while-revalidate pattern
- [x] Implement TTL-based cache expiration
- [x] Add memory leak prevention
- [ ] Prefetch critical API endpoints
- [ ] Implement API request batching
- [ ] Monitor cache stats in development
- [ ] Test cache behavior with DevTools Network tab
- [ ] Add retry logic for failed requests

## Phase 6: Build Optimization ✓

- [x] Configure Terser minification
- [x] Enable CSS code splitting
- [x] Disable sourcemaps in production
- [x] Configure chunk naming for caching
- [x] Add hash-based asset names
- [x] Set up asset fingerprinting
- [ ] Test production build locally
- [ ] Verify source map absence in production
- [ ] Check console output for warnings
- [ ] Monitor build size over time

## Phase 7: Asset Caching ✓

- [x] Create _headers file for cache control
- [x] Configure long-term caching for hashed assets
- [x] Set cache-control for HTML (no cache)
- [x] Set cache-control for API (no cache)
- [x] Add security headers
- [x] Enable CORS for fonts
- [ ] Deploy and verify headers
- [ ] Test cache behavior with DevTools
- [ ] Verify cache-busting works correctly
- [ ] Monitor cache hit rates

## Phase 8: Web Vitals Monitoring ✓

- [x] Create web vitals tracking module
- [x] Implement LCP monitoring
- [x] Implement INP monitoring
- [x] Implement CLS monitoring
- [x] Send metrics to analytics
- [x] Initialize tracking in main.tsx
- [ ] Set up Google Analytics goals
- [ ] Create dashboard for tracking metrics
- [ ] Set up alerts for regressions
- [ ] Monitor real user metrics (RUM)

## Phase 9: Lighthouse CI ✓

- [x] Create .lighthouserc.js configuration
- [x] Set performance score targets (90+)
- [x] Set accessibility targets (95+)
- [x] Set Core Web Vitals thresholds
- [ ] Integrate with CI/CD pipeline
- [ ] Configure automated testing
- [ ] Set up result reporting
- [ ] Create performance budget alerts
- [ ] Generate historical reports

## Phase 10: Documentation ✓

- [x] Create PERFORMANCE.md guide
- [x] Create PERFORMANCE_METRICS.md
- [x] Create verification script
- [x] Document optimization strategies
- [x] Add code examples
- [x] Include troubleshooting guide
- [x] Create this checklist
- [ ] Add inline code comments
- [ ] Create video tutorial
- [ ] Document for team onboarding

## Ongoing Maintenance

### Weekly
- [ ] Monitor Core Web Vitals in Google Analytics
- [ ] Check Lighthouse CI results
- [ ] Review error logs for performance issues

### Monthly
- [ ] Run full Lighthouse audit
- [ ] Review bundle analysis (stats.html)
- [ ] Check for new performance opportunities
- [ ] Update dependencies for performance improvements

### Quarterly
- [ ] Comprehensive performance audit
- [ ] User experience metrics review
- [ ] Competitive analysis
- [ ] Plan next optimization phase

## Component Implementation Guide

### 1. Use OptimizedImage for All Images
```tsx
import OptimizedImage from '@/components/OptimizedImage';

// For hero/critical images
<OptimizedImage
  src={jpegImage}
  srcWebp={webpImage}
  srcAvif={avifImage}
  alt="Description"
  width={1200}
  height={800}
  priority={true}
  sizes="(max-width: 640px) 100vw, 80vw"
/>

// For below-fold images
<OptimizedImage
  src={jpegImage}
  srcWebp={webpImage}
  srcAvif={avifImage}
  alt="Description"
  width={600}
  height={400}
  placeholder={blurImage}
  lazy={true}
/>
```

### 2. Optimize List Components
```tsx
import { memoComponent } from '@/hooks/useMemoized';
import { useCallback, useMemo } from 'react';

// Memoize list items
const ProductCardMemo = memoComponent(ProductCard);

// Parent component
const ProductList = ({ products, onSelect }) => {
  // Memoize handler
  const handleSelect = useCallback((productId) => {
    onSelect(productId);
  }, [onSelect]);

  // Memoize filtered data
  const filteredProducts = useMemo(() => {
    return filterProducts(products);
  }, [products]);

  return (
    <div>
      {filteredProducts.map(product => (
        <ProductCardMemo
          key={product.id}
          product={product}
          onSelect={handleSelect}
        />
      ))}
    </div>
  );
};
```

### 3. Optimize Search Input
```tsx
import { useDebounce } from '@/hooks/usePerformanceOptimization';

const SearchProducts = ({ onSearch }) => {
  const [value, setValue] = useState('');

  // Debounce expensive search
  const debouncedSearch = useDebounce(
    (query) => onSearch(query),
    300,
    [onSearch]
  );

  const handleChange = (e) => {
    setValue(e.target.value);
    debouncedSearch(e.target.value);
  };

  return <input value={value} onChange={handleChange} />;
};
```

### 4. Optimize Scroll Handlers
```tsx
import { useThrottle } from '@/hooks/usePerformanceOptimization';

const InfiniteScroll = ({ onLoadMore }) => {
  const throttledScroll = useThrottle(
    () => {
      const { scrollY } = window;
      if (scrollY + window.innerHeight >= document.body.scrollHeight) {
        onLoadMore();
      }
    },
    200,
    [onLoadMore]
  );

  useEffect(() => {
    window.addEventListener('scroll', throttledScroll);
    return () => window.removeEventListener('scroll', throttledScroll);
  }, [throttledScroll]);

  return <div>Content with infinite scroll</div>;
};
```

### 5. Use API Caching
```tsx
import { apiCache, prefetchCriticalData } from '@/lib/api-cache';

// In component
const [products, setProducts] = useState(null);

useEffect(() => {
  // Fetch with 5-minute cache
  apiCache
    .fetch('/api/products', {
      cache: { ttl: 5 * 60 * 1000 }
    })
    .then(setProducts)
    .catch(console.error);
}, []);

// Prefetch on app load
useEffect(() => {
  prefetchCriticalData([
    { url: '/api/categories', priority: 'high' },
    { url: '/api/featured', priority: 'high' },
    { url: '/api/recommendations', priority: 'low' }
  ]);
}, []);
```

## Verification

Run verification checklist:
```bash
npm run perf:verify
```

Should output:
```
✓ OptimizedImage component exists
✓ Font preload configured
✓ Web Vitals module exists
✓ API cache module exists
✓ Vite optimized
✓ Build optimization enabled
✓ Bundle visualizer configured
✓ Performance hooks available
✓ Memoization utilities available
✓ Lighthouse config exists
✓ Caching headers configured
✓ Performance guide exists
✓ visualizer installed

Results: 13/13 checks passed

🎉 All optimizations verified!
```

## Testing on Different Devices/Networks

### Mobile (Slow 4G)
```bash
npm run build && npm run preview
# Open DevTools → Network → Slow 4G
# Refresh page and observe metrics
```

### Slow CPU
```bash
# Open DevTools → Performance tab
# Check CPU Throttling: 6x
# Record page load and analyze
```

### Lighthouse Audit
```bash
# Full audit with recommendations
lighthouse https://liveiticonic.com --view
```

## Performance Regression Prevention

Add to pre-commit hook:
```bash
#!/bin/bash
npm run perf:verify || exit 1
npm run build || exit 1
# Optional: npm run perf:lighthouse
```

## Success Metrics

- ✓ All Core Web Vitals met (LCP ≤2.5s, INP <200ms, CLS <0.1)
- ✓ Lighthouse Performance ≥ 90
- ✓ Bundle size < 200KB (gzipped)
- ✓ First paint < 2 seconds
- ✓ Repeat visit loads instantly
- ✓ Mobile (slow 4G) still acceptable
- ✓ Zero layout shifts
- ✓ Smooth 60fps interactions
- ✓ All images optimized
- ✓ Fonts don't block rendering

---

**Last Updated**: 2025-11-12
**Status**: Phase 1-10 Complete (Initial Implementation)
**Next Phase**: Continuous Monitoring & Maintenance
