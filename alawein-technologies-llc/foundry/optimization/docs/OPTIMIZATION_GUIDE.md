# Performance Optimization Guide

## Table of Contents

1. [Overview](#overview)
2. [Bundle Optimization](#bundle-optimization)
3. [Performance Monitoring](#performance-monitoring)
4. [Caching Strategy](#caching-strategy)
5. [Continuous Improvement](#continuous-improvement)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This comprehensive performance optimization system provides automated tools and strategies to ensure all three applications (Scientific Tinder, Chaos Engine, Ghost Researcher) achieve and maintain optimal performance.

### Target Metrics

- **Lighthouse Score:** 90+
- **LCP (Largest Contentful Paint):** <2.5s
- **FID (First Input Delay):** <100ms
- **CLS (Cumulative Layout Shift):** <0.1
- **Bundle Size:** <250KB (gzipped)
- **API Response Time:** <200ms
- **Cache Hit Rate:** >80%

---

## Bundle Optimization

### Configuration Files

#### 1. Webpack Analyzer Configuration
Location: `/optimization/bundle-analysis/webpack-analyzer.config.js`

This configuration provides:
- Automatic bundle analysis
- Code splitting strategies
- Tree shaking optimization
- Compression (gzip & brotli)

#### 2. Next.js Optimized Configuration
Location: `/optimization/bundle-analysis/next.config.optimized.js`

Features:
- SWC minification
- Image optimization
- PWA support
- Optimized package imports
- Cache headers

### Running Bundle Analysis

```bash
# Analyze a specific app
cd frontend/scientific-tinder
ANALYZE=true npm run build

# Run comprehensive optimization
cd /home/user/Foundry/optimization/scripts
./optimize.sh
```

### Code Splitting Strategy

```typescript
// Dynamic imports for heavy components
const HeavyChart = dynamic(() => import('@/components/HeavyChart'), {
  loading: () => <ChartSkeleton />,
  ssr: false
});

// Route-based splitting (automatic in Next.js)
// Each page is automatically code-split

// Conditional loading
if (userWantsAdvancedFeatures) {
  const { AdvancedFeature } = await import('./AdvancedFeature');
}
```

### Tree Shaking Optimization

```typescript
// ❌ Bad - imports entire library
import _ from 'lodash';

// ✅ Good - imports only what's needed
import debounce from 'lodash/debounce';

// ❌ Bad - imports all icons
import * as Icons from 'lucide-react';

// ✅ Good - imports specific icons
import { Search, User, Settings } from 'lucide-react';
```

---

## Performance Monitoring

### Core Web Vitals Implementation

Location: `/optimization/monitoring/web-vitals.ts`

```typescript
// Initialize in _app.tsx
import { initWebVitals } from '@/optimization/monitoring/web-vitals';

export function reportWebVitals(metric: NextWebVitalsMetric) {
  // Metrics are automatically sent to analytics
  initWebVitals();
}
```

### Performance Dashboard

Location: `/optimization/monitoring/PerformanceDashboard.tsx`

Access the dashboard at: `/admin/performance`

Features:
- Real-time Core Web Vitals
- Bundle size tracking
- Cache performance metrics
- API performance monitoring
- Lighthouse scores

### Custom Metrics Tracking

```typescript
import { customMetrics } from '@/optimization/monitoring/web-vitals';

// Track component render time
const ComponentWithTracking = () => {
  useEffect(() => {
    const start = performance.now();

    return () => {
      const renderTime = performance.now() - start;
      customMetrics.trackComponentRender('ComponentName', renderTime);
    };
  }, []);
};

// Track API performance
const fetchData = async () => {
  const start = performance.now();

  try {
    const response = await fetch('/api/data');
    const duration = performance.now() - start;
    customMetrics.trackAPICall('/api/data', duration, response.status);
    return response.json();
  } catch (error) {
    // Handle error
  }
};
```

---

## Caching Strategy

### Multi-Layer Caching Architecture

Location: `/optimization/caching/cache-config.ts`

#### Layer 1: Browser Cache
- Static assets: 1 year
- Images: 30 days
- Fonts: 1 year
- HTML: 1 hour

#### Layer 2: Service Worker
- Offline support
- Background sync
- Cache strategies per route

#### Layer 3: React Query (Memory)
- User data: 5 minutes stale time
- Research papers: 24 hours
- Matches: 10 minutes
- Ideas: 15 minutes

#### Layer 4: CDN
- Static assets
- API responses (read-only)

### Service Worker Setup

Location: `/optimization/caching/service-worker.js`

Register in your app:
```typescript
// In _app.tsx or layout.tsx
useEffect(() => {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker
      .register('/service-worker.js')
      .then(reg => console.log('Service Worker registered'))
      .catch(err => console.error('Service Worker registration failed'));
  }
}, []);
```

### Cache Invalidation

```typescript
import { cacheInvalidationManager } from '@/optimization/caching/cache-config';

// Invalidate on user actions
const handleUserUpdate = async () => {
  await updateUser(data);
  await cacheInvalidationManager.invalidate('user-update');
};

// Clear all caches
const handleLogout = async () => {
  await logout();
  await cacheInvalidationManager.clearAll();
};

// Get cache statistics
const stats = await cacheInvalidationManager.getCacheStats();
console.log('Cache stats:', stats);
```

### React Query Configuration

```typescript
import { createQueryClient } from '@/optimization/caching/cache-config';

const queryClient = createQueryClient();

// Use in your app
<QueryClientProvider client={queryClient}>
  <App />
</QueryClientProvider>

// Optimized query with caching
const { data, isLoading } = useQuery({
  queryKey: ['papers', filters],
  queryFn: fetchPapers,
  staleTime: 60 * 60 * 1000, // 1 hour
  gcTime: 24 * 60 * 60 * 1000, // 24 hours
});
```

---

## Continuous Improvement

### Automated Performance Audits

Location: `/optimization/continuous-improvement/performance-audit.ts`

#### Configuration

```typescript
const auditConfig = {
  schedule: 'weekly',
  applications: ['scientific-tinder', 'chaos-engine', 'ghost-researcher'],
  metrics: ['lighthouse', 'bundle-size', 'lcp', 'fid', 'cls'],
  thresholds: {
    'lighthouse-performance': 90,
    'bundle-size': 500,
    'lcp': 2500,
    'fid': 100,
    'cls': 0.1
  },
  notifications: {
    email: ['team@Foundry.com'],
    slack: '#performance-alerts'
  }
};
```

#### Running Audits

```bash
# Manual audit
npm run audit:performance

# Schedule automated audits
npm run audit:schedule
```

### Performance Improvement Workflow

1. **Weekly Audit**
   - Automated performance audit runs
   - Issues and recommendations generated
   - Reports saved to `/optimization/continuous-improvement/reports/`

2. **Review & Prioritize**
   - Team reviews audit results
   - Prioritize issues by severity
   - Create tasks for improvements

3. **Implement Optimizations**
   - Follow recommendations
   - Test improvements locally
   - Measure impact

4. **Validate & Deploy**
   - Run performance tests
   - Ensure no regressions
   - Deploy to production

5. **Monitor & Iterate**
   - Track metrics in production
   - Compare with baseline
   - Iterate as needed

---

## Best Practices

### React Performance

```typescript
// 1. Memoization
const ExpensiveComponent = React.memo(({ data }) => {
  // Component only re-renders when data changes
});

// 2. useMemo for expensive calculations
const expensiveValue = useMemo(() => {
  return calculateExpensiveValue(data);
}, [data]);

// 3. useCallback for stable references
const handleClick = useCallback(() => {
  doSomething(id);
}, [id]);

// 4. Lazy loading
const LazyComponent = lazy(() => import('./LazyComponent'));

// 5. Virtual scrolling for large lists
import { FixedSizeList } from 'react-window';

<FixedSizeList
  height={600}
  itemCount={1000}
  itemSize={50}
  width="100%"
>
  {Row}
</FixedSizeList>
```

### Image Optimization

```typescript
// Use next/image for automatic optimization
import Image from 'next/image';

<Image
  src="/image.jpg"
  alt="Description"
  width={800}
  height={600}
  priority={false} // true for above-the-fold images
  placeholder="blur"
  blurDataURL={blurDataUrl}
/>

// Responsive images
<Image
  src="/image.jpg"
  alt="Description"
  sizes="(max-width: 640px) 100vw,
         (max-width: 1024px) 50vw,
         33vw"
  fill
/>
```

### API Optimization

```typescript
// 1. Request batching
const batchRequests = async (requests: Request[]) => {
  return Promise.all(requests.map(req => fetch(req)));
};

// 2. Debounced search
const debouncedSearch = useMemo(
  () => debounce(search, 300),
  []
);

// 3. Pagination
const { data } = useInfiniteQuery({
  queryKey: ['items'],
  queryFn: ({ pageParam = 0 }) => fetchItems(pageParam),
  getNextPageParam: (lastPage) => lastPage.nextCursor,
});

// 4. Field selection (GraphQL)
const OPTIMIZED_QUERY = gql`
  query GetUser($id: ID!) {
    user(id: $id) {
      id
      name
      # Only request fields you need
    }
  }
`;
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Large Bundle Size

**Symptoms:**
- Slow initial load
- Bundle size >500KB

**Solutions:**
```bash
# Analyze bundle
ANALYZE=true npm run build

# Find large dependencies
npm ls --depth=0 | grep -E "^[├└]" | sort -k2 -hr

# Remove unused dependencies
npx depcheck

# Use dynamic imports
const Heavy = dynamic(() => import('./Heavy'));
```

#### 2. Poor LCP Score

**Symptoms:**
- LCP >4s
- Slow server response

**Solutions:**
```typescript
// Preload critical resources
<link rel="preload" href="/font.woff2" as="font" crossOrigin="" />

// Optimize server response
export async function getStaticProps() {
  // Use ISR for better performance
  return {
    props: { data },
    revalidate: 60 // Revalidate every 60 seconds
  };
}

// Use CDN for images
<Image src="https://cdn.example.com/image.jpg" />
```

#### 3. High CLS Score

**Symptoms:**
- Layout shifts during load
- CLS >0.25

**Solutions:**
```css
/* Reserve space for dynamic content */
.image-container {
  aspect-ratio: 16 / 9;
  width: 100%;
}

/* Use CSS containment */
.ad-container {
  contain: layout style paint;
  min-height: 250px;
}
```

#### 4. Memory Leaks

**Symptoms:**
- Increasing memory usage
- Page becomes slow over time

**Solutions:**
```typescript
// Clean up subscriptions
useEffect(() => {
  const subscription = subscribe();
  return () => subscription.unsubscribe();
}, []);

// Clear timers
useEffect(() => {
  const timer = setTimeout(() => {}, 1000);
  return () => clearTimeout(timer);
}, []);

// Remove event listeners
useEffect(() => {
  const handler = () => {};
  window.addEventListener('resize', handler);
  return () => window.removeEventListener('resize', handler);
}, []);
```

### Performance Debugging Tools

```bash
# Chrome DevTools Performance Panel
1. Open DevTools (F12)
2. Go to Performance tab
3. Start recording
4. Perform actions
5. Stop and analyze

# React DevTools Profiler
1. Install React DevTools extension
2. Go to Profiler tab
3. Start profiling
4. Interact with app
5. Analyze render times

# Lighthouse CLI
npx lighthouse https://localhost:3000 \
  --view \
  --chrome-flags="--headless"

# Bundle analyzer
ANALYZE=true npm run build
```

---

## Monitoring Checklist

### Daily Checks
- [ ] Core Web Vitals within thresholds
- [ ] No new error spikes
- [ ] API response times <200ms
- [ ] Cache hit rate >80%

### Weekly Checks
- [ ] Run performance audit
- [ ] Review bundle size changes
- [ ] Check for unused dependencies
- [ ] Analyze slow queries
- [ ] Review error logs

### Monthly Checks
- [ ] Full dependency audit
- [ ] Security vulnerability scan
- [ ] Performance trend analysis
- [ ] User feedback review
- [ ] Infrastructure optimization

---

## Resources

### Documentation
- [Web Vitals](https://web.dev/vitals/)
- [Next.js Performance](https://nextjs.org/docs/advanced-features/measuring-performance)
- [React Performance](https://react.dev/learn/render-and-commit)

### Tools
- [Lighthouse CI](https://github.com/GoogleChrome/lighthouse-ci)
- [Bundle Analyzer](https://github.com/webpack-contrib/webpack-bundle-analyzer)
- [WebPageTest](https://www.webpagetest.org/)

### Monitoring Services
- [Google PageSpeed Insights](https://pagespeed.web.dev/)
- [GTmetrix](https://gtmetrix.com/)
- [Calibre](https://calibreapp.com/)

---

## Support

For questions or issues related to performance optimization:

1. Check this documentation
2. Review audit reports in `/optimization/continuous-improvement/reports/`
3. Check performance dashboard at `/admin/performance`
4. Contact the performance team

Remember: **Performance is a feature, not an afterthought!**