# Analytics & Tracking Infrastructure Documentation

## Table of Contents

1. [Overview](#overview)
2. [Setup Instructions](#setup-instructions)
3. [Core Components](#core-components)
4. [Implementation Guide](#implementation-guide)
5. [Event Tracking](#event-tracking)
6. [Performance Monitoring](#performance-monitoring)
7. [Error Tracking](#error-tracking)
8. [Consent Management](#consent-management)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Testing](#testing)

---

## Overview

The analytics infrastructure provides comprehensive tracking capabilities including:

- **Google Analytics 4 Integration** - Full GA4 support with event validation
- **Performance Monitoring** - Core Web Vitals tracking (LCP, FID, CLS, TTI, INP)
- **Error Tracking** - JavaScript errors, API errors, and session replay triggers
- **Consent Management** - GDPR/CCPA compliant consent tracking
- **User Behavior Analytics** - Click tracking, engagement metrics, time on page
- **Scroll Depth Tracking** - Milestone-based scroll tracking (25%, 50%, 75%, 100%)
- **E-commerce Events** - Product views, cart operations, purchases, refunds

### Key Features

✅ **Type-Safe** - Full TypeScript support with comprehensive type definitions
✅ **Privacy-First** - No PII collection, GDPR/CCPA compliant
✅ **Consent-Based** - Respects user privacy preferences
✅ **Debug Mode** - Development-friendly logging and debugging
✅ **Multi-Provider** - Extensible architecture for multiple analytics platforms
✅ **Performance** - Minimal performance impact with optimized tracking

---

## Setup Instructions

### 1. Environment Configuration

Add the following to your `.env` file:

```env
# Google Analytics 4 Measurement ID
VITE_GA_MEASUREMENT_ID=G-XXXXXXXXXX

# Google Tag Manager ID (optional)
VITE_GTM_ID=GTM-XXXXXXX

# Plausible Analytics Domain (alternative to GA4)
VITE_PLAUSIBLE_DOMAIN=yourdomain.com

# Enable analytics debug mode
VITE_ANALYTICS_DEBUG=false
```

### 2. Google Analytics 4 Setup

1. Go to [Google Analytics](https://analytics.google.com/)
2. Create a new property or use existing one
3. In Admin → Property Settings, find your **Measurement ID** (format: `G-XXXXXXXXXX`)
4. Add to `.env` as `VITE_GA_MEASUREMENT_ID`

### 3. Data Stream Configuration

1. In Google Analytics, go to Admin → Data Streams
2. Select your web stream
3. Configure under "Measurement Settings":
   - Enable "Enhanced measurement" for automatic event tracking
   - Enable "Consent Mode" for GDPR compliance

### 4. Initialize Analytics in App

The analytics system auto-initializes on app load, but you can manually initialize:

```typescript
import { initializeAnalytics } from '@/lib/analytics';

// Initialize with config
initializeAnalytics({
  ga4MeasurementId: import.meta.env.VITE_GA_MEASUREMENT_ID,
  debugMode: import.meta.env.DEV,
  consentRequired: true,
});
```

---

## Core Components

### 1. Analytics Manager (`src/lib/analytics.ts`)

Central analytics engine with support for multiple providers.

**Key Classes:**
- `AnalyticsManager` - Main analytics class
- `GA4Provider` - Google Analytics 4 provider
- `DebugProvider` - Development console logging

**Usage:**
```typescript
import { getAnalytics } from '@/lib/analytics';

const analytics = getAnalytics();

// Track events
analytics.track('custom_event', { property: 'value' });

// E-commerce events
analytics.purchase({
  transaction_id: '123',
  value: 99.99,
  currency: 'USD',
  items: [...]
});
```

### 2. Performance Monitor (`src/lib/performance.ts`)

Tracks Core Web Vitals and performance metrics.

**Tracked Metrics:**
- **LCP** (Largest Contentful Paint) - Page load performance
- **FID** (First Input Delay) - Interactivity
- **CLS** (Cumulative Layout Shift) - Visual stability
- **TTI** (Time to Interactive) - When page becomes interactive
- **INP** (Interaction to Next Paint) - Event processing time

**Usage:**
```typescript
import { getPerformanceMonitor } from '@/lib/performance.ts';

const perf = getPerformanceMonitor();

// Get all metrics
const vitals = perf.getWebVitals();

// Get specific metric
const lcp = perf.getMetric('lcp');

// Custom performance marks
perf.mark('operation_start');
// ... perform operation ...
perf.measure('operation_time', 'operation_start');
```

### 3. Error Tracking (`src/lib/errorTracking.ts`)

Comprehensive error capture and monitoring.

**Features:**
- Global error handler
- Unhandled promise rejection tracking
- API error monitoring
- Session replay triggers
- Breadcrumb trail for context

**Usage:**
```typescript
import { getErrorTracker } from '@/lib/errorTracking';

const errorTracker = getErrorTracker();

// Manual error capture
errorTracker.captureError({
  type: 'api_error',
  message: 'Failed to fetch user data',
  context: { endpoint: '/api/user' }
});

// Add breadcrumbs for context
errorTracker.addBreadcrumb('User clicked button', 'user-action');
errorTracker.addApiCallBreadcrumb('GET', '/api/data', 200);

// Get error report
const report = errorTracker.exportErrorReport();
```

### 4. Consent Manager (`src/lib/consentManager.ts`)

GDPR/CCPA compliant consent management.

**Categories:**
- `analytics` - Analytics tracking
- `marketing` - Marketing cookies
- `functional` - Required for functionality
- `performance` - Performance monitoring
- `preferences` - User preferences

**Usage:**
```typescript
import { getConsentManager } from '@/lib/consentManager';

const consent = getConsentManager();

// Check consent
if (consent.hasConsent('analytics')) {
  // Track analytics
}

// Handle consent changes
consent.onConsentChange('analytics', (hasConsent) => {
  console.log('Analytics consent:', hasConsent);
});

// Set consent
consent.setConsent('marketing', true);

// Get compliance report
const report = consent.getComplianceReport();
```

---

## Implementation Guide

### In React Components

#### Basic Setup

```typescript
import { useAnalytics } from '@/hooks/useAnalytics';
import { useScrollDepth } from '@/hooks/useScrollDepth';
import { useUserBehavior } from '@/hooks/useUserBehavior';

export const MyComponent = () => {
  const analytics = useAnalytics();
  const scrollDepth = useScrollDepth();
  const behavior = useUserBehavior();

  return (
    <div>
      {/* Component JSX */}
    </div>
  );
};
```

#### Product Page Tracking

```typescript
export const ProductPage = ({ productId }) => {
  const { trackProductView, trackAddToCart } = useAnalytics();

  useEffect(() => {
    // Track product view
    trackProductView({
      id: productId,
      name: product.name,
      price: product.price,
      category: product.category,
      currency: 'USD',
      quantity: 1,
    });
  }, [productId]);

  const handleAddToCart = (quantity) => {
    trackAddToCart({
      id: productId,
      name: product.name,
      price: product.price,
      quantity,
      currency: 'USD',
    });
  };

  return (
    <button onClick={() => handleAddToCart(1)}>
      Add to Cart
    </button>
  );
};
```

#### Checkout Tracking

```typescript
export const CheckoutPage = () => {
  const {
    trackCheckoutStarted,
    trackCheckoutStep,
    trackPurchase
  } = useAnalytics();

  const handleCheckoutStart = () => {
    trackCheckoutStarted(cartItems, total);
  };

  const handleStepChange = (step, stepNumber) => {
    trackCheckoutStep(step, stepNumber);
  };

  const handlePurchaseComplete = (orderId, items) => {
    trackPurchase(
      orderId,
      total,
      items,
      'USD',
      tax,
      shipping,
      coupon
    );
  };

  return (/* ... */);
};
```

#### Button & Link Tracking

```typescript
export const CTAButton = ({ label }) => {
  const { trackButtonClick } = useAnalytics();

  return (
    <button onClick={() => trackButtonClick(label)}>
      {label}
    </button>
  );
};

export const ExternalLink = ({ href, text }) => {
  const { trackLinkClick } = useAnalytics();

  return (
    <a
      href={href}
      onClick={() => trackLinkClick(text, href, true)}
      target="_blank"
      rel="noopener noreferrer"
    >
      {text}
    </a>
  );
};
```

---

## Event Tracking

### Standard Events

#### Page View
```typescript
analytics.pageView('/shop', 'Shop');
```

#### E-commerce Events

```typescript
// View product
analytics.viewItem({
  id: '123',
  name: 'Premium T-Shirt',
  price: 29.99,
  quantity: 1,
  category: 'Apparel',
  brand: 'Live It Iconic'
});

// Add to cart
analytics.addToCart({
  id: '123',
  name: 'Premium T-Shirt',
  price: 29.99,
  quantity: 1
});

// Begin checkout
analytics.beginCheckout(items, total);

// Complete purchase
analytics.purchase({
  transaction_id: 'order_123',
  value: 99.99,
  currency: 'USD',
  tax: 8.00,
  shipping: 5.00,
  items: [...]
});
```

#### User Engagement

```typescript
// Search
analytics.search('luxury apparel');

// Form submission
analytics.formSubmit('contact_form', {
  form_type: 'contact',
  field_count: 4
});

// Social share
analytics.socialShare('instagram', 'product', 'Premium T-Shirt');

// File download
analytics.fileDownload('brand-guidelines.pdf', 'pdf');

// User authentication
analytics.login('email');
analytics.logout();
```

### Custom Events

```typescript
analytics.track('custom_event_name', {
  property1: 'value1',
  property2: 123,
  property3: true
});
```

### Event Naming Conventions

Use snake_case for all event names:

```typescript
// ✅ Good
analytics.track('user_signup_email');

// ❌ Bad
analytics.track('userSignupEmail'); // camelCase
analytics.track('User Signup Email'); // spaces
```

**Maximum 40 characters** per event name.

---

## Performance Monitoring

### Core Web Vitals

Access metrics after page load:

```typescript
import { getPerformanceMonitor } from '@/lib/performance';

const perf = getPerformanceMonitor();

// Get all vitals
const vitals = perf.getWebVitals();
console.log('LCP:', vitals.lcp?.value, 'ms');
console.log('FID:', vitals.fid?.value, 'ms');
console.log('CLS:', vitals.cls?.value);

// Get all metrics
const allMetrics = perf.getAllMetrics();
```

### API Performance Tracking

```typescript
const startTime = performance.now();

try {
  const response = await fetch('/api/endpoint');
  const duration = performance.now() - startTime;

  perf.trackApiCall(
    '/api/endpoint',
    'GET',
    duration,
    response.status,
    response.headers.get('content-length')
  );
} catch (error) {
  const duration = performance.now() - startTime;
  perf.trackApiCall('/api/endpoint', 'GET', duration, 0);
}
```

### Custom Performance Marks

```typescript
// Mark operation start
perf.mark('data_fetch');

// Perform operation
const data = await fetchData();

// Measure operation
const duration = perf.measure('fetch_time', 'data_fetch');
console.log(`Data fetch took ${duration}ms`);
```

---

## Error Tracking

### Automatic Error Capture

Automatically captured:
- JavaScript errors
- Unhandled promise rejections
- Network errors
- Console errors/warnings

### Manual Error Capture

```typescript
const errorTracker = getErrorTracker();

errorTracker.captureError({
  type: 'api_error',
  message: 'Failed to fetch user profile',
  severity: 'high',
  context: {
    endpoint: '/api/user',
    userId: '123'
  }
});
```

### Breadcrumbs for Context

```typescript
// User actions
errorTracker.addUserActionBreadcrumb('Clicked checkout button');

// Navigation
errorTracker.addNavigationBreadcrumb('/shop', '/product/123');

// API calls
errorTracker.addApiCallBreadcrumb('POST', '/api/orders', 201);

// Custom
errorTracker.addBreadcrumb('Payment processed', 'payment');
```

### Error Reports

```typescript
// Get all errors
const errors = errorTracker.getErrorLogs();

// Get errors by type
const apiErrors = errorTracker.getErrorsByType('api_error');

// Get critical errors
const critical = errorTracker.getErrorsByType('critical');

// Export error report
const report = errorTracker.exportErrorReport();
console.log(report);
```

---

## Consent Management

### Initialization

Consent is automatically initialized on app load, detecting user jurisdiction:

- **EU** - GDPR applies (explicit opt-in required)
- **Canada** - PIPEDA applies
- **US/Other** - Can use implied consent

### User Consent Settings

```typescript
import { getConsentManager } from '@/lib/consentManager';

const consent = getConsentManager();

// Check specific consent
if (consent.hasConsent('analytics')) {
  // Run analytics
}

// Set consent
consent.setConsent('marketing', true);

// Accept/Reject all
consent.acceptAll();
consent.rejectAll();

// Get current settings
const settings = consent.getConsent();
console.log(settings); // { analytics: true, marketing: false, ... }
```

### Consent Banner Implementation

```typescript
import { getConsentManager } from '@/lib/consentManager';

export const ConsentBanner = () => {
  const consent = getConsentManager();

  if (!consent.shouldShowBanner()) {
    return null; // User already made choice
  }

  const handleAcceptAll = () => {
    consent.acceptAll();
    consent.markConsentBannerShown();
  };

  const handleRejectAll = () => {
    consent.rejectAll();
    consent.markConsentBannerShown();
  };

  return (
    <div className="consent-banner">
      <p>We use cookies to improve your experience.</p>
      <button onClick={handleAcceptAll}>Accept All</button>
      <button onClick={handleRejectAll}>Reject All</button>
    </div>
  );
};
```

### Category Descriptions

```typescript
const consent = getConsentManager();

// Get all category descriptions
const descriptions = consent.getCategoryDescriptions();
// {
//   analytics: "Help us understand how you use our website...",
//   marketing: "Allow us to show you relevant ads...",
//   ...
// }

// Get specific category description
const analyticsDesc = consent.getCategoryDescription('analytics');
```

### Data Subject Rights

```typescript
// GDPR: Right to access
const consentData = consent.exportConsentData();

// GDPR: Right to delete
consent.deleteConsentData();

// GDPR: Right to rectification
consent.updateConsentData({
  analytics: false,
  marketing: true
});
```

---

## Best Practices

### 1. No PII Collection

✅ **DO:**
```typescript
// Good - aggregated, no PII
analytics.track('user_segment_event', {
  user_tier: 'premium',
  purchase_count: 5
});
```

❌ **DON'T:**
```typescript
// Bad - contains PII
analytics.track('user_data', {
  email: 'user@example.com',
  phone: '555-1234',
  address: '123 Main St'
});
```

### 2. Consent Checking

Always check consent before tracking sensitive events:

```typescript
const consent = getConsentManager();

if (consent.hasConsent('marketing')) {
  // Track marketing event
}
```

### 3. Event Validation

Use the provided configurations for consistent naming:

```typescript
import { EventNames } from '@/config/analytics';

// Use constants instead of strings
analytics.track(EventNames.PAGE_VIEW);
analytics.track(EventNames.ADD_TO_CART);
```

### 4. Performance Considerations

- Track important events only
- Batch events when possible
- Avoid excessive event tracking
- Use appropriate throttling for scroll events

### 5. Testing Analytics

Enable debug mode during development:

```env
VITE_ANALYTICS_DEBUG=true
```

Check browser console for detailed event logs.

### 6. Error Handling

Add breadcrumbs before errors occur:

```typescript
const errorTracker = getErrorTracker();

try {
  errorTracker.addBreadcrumb('Starting user signup', 'user');
  await signupUser(data);
  errorTracker.addBreadcrumb('Signup successful', 'user');
} catch (error) {
  errorTracker.captureError({
    type: 'validation_error',
    message: error.message
  });
}
```

---

## Troubleshooting

### Events Not Appearing in GA4

1. **Check Configuration:**
   ```typescript
   const analytics = getAnalytics();
   console.log(analytics.getConfig());
   ```

2. **Enable Debug Mode:**
   ```env
   VITE_ANALYTICS_DEBUG=true
   ```

3. **Verify Measurement ID:**
   - Ensure `VITE_GA_MEASUREMENT_ID` is correct
   - Format: `G-XXXXXXXXXX`

4. **Check Consent:**
   ```typescript
   const consent = getConsentManager();
   console.log(consent.getConsent());
   ```

5. **Inspect GA4 Settings:**
   - Go to Google Analytics → Admin → Data Streams
   - Verify stream is active
   - Check for IP filters blocking localhost

### Events Blocked by Consent

```typescript
const consent = getConsentManager();

// User must consent to analytics
if (!consent.hasConsent('analytics')) {
  console.log('Analytics disabled by user');
}
```

### Performance Issues

Check if tracking is causing performance problems:

```typescript
const perf = getPerformanceMonitor();

// Monitor analytics-related operations
perf.mark('analytics_event');
analytics.track('event');
perf.measure('event_timing', 'analytics_event');

// Review metrics
const metrics = perf.getAllMetrics();
```

### Consent Banner Not Showing

```typescript
const consent = getConsentManager();

// Check banner status
console.log('Should show:', consent.shouldShowBanner());
console.log('Was shown:', consent.wasBannerShown());

// Reset consent to test banner
consent.clearAll();
```

---

## Testing

### Unit Testing Analytics

```typescript
import { getAnalytics } from '@/lib/analytics';
import { getConsentManager } from '@/lib/consentManager';

describe('Analytics', () => {
  it('should track events when consent is given', () => {
    const consent = getConsentManager();
    consent.setConsent('analytics', true);

    const analytics = getAnalytics();
    analytics.track('test_event', { test: true });

    // Verify event was tracked
    const events = analytics.getEventHistory();
    expect(events.some(e => e.name === 'test_event')).toBe(true);
  });

  it('should respect consent settings', () => {
    const consent = getConsentManager();
    consent.setConsent('analytics', false);

    // Event should still be tracked but not sent
    // (implementation depends on your tracking layer)
  });
});
```

### Manual Testing

1. **Debug Mode:**
   ```env
   VITE_ANALYTICS_DEBUG=true
   ```
   Watch console for event logging

2. **GA4 Real-Time Reports:**
   - Open Google Analytics
   - Go to Reports → Real-time
   - Perform actions on site
   - Verify events appear

3. **LocalStorage Inspection:**
   ```javascript
   // Events stored in localStorage
   JSON.parse(localStorage.getItem('analytics_events'))

   // Consent settings
   JSON.parse(localStorage.getItem('user_consent_settings'))

   // Error logs
   JSON.parse(localStorage.getItem('error_logs'))
   ```

---

## API Reference

### Analytics Manager

```typescript
// Event tracking
track(eventName: string, properties?: AnalyticsProperties)
pageView(path: string, title?: string)
trackScrollDepth(depth: number)
trackTimeOnPage(pagePath: string, secondsOnPage: number)

// E-commerce
viewItem(product: EcommerceProduct)
addToCart(product: EcommerceProduct)
removeFromCart(product: EcommerceProduct)
purchase(purchase: PurchaseData)
refund(transactionId: string, value: number, currency?: string)

// User engagement
search(query: string)
buttonClick(buttonName: string, properties?: AnalyticsProperties)
linkClick(linkName: string, url?: string, isExternalLink?: boolean)
socialShare(platform: string, contentType?: string, contentTitle?: string)
formSubmit(formName: string, fields?: Record<string, string>)

// User identification
identify(userId: string, userProperties?: UserProperties)
setUserId(userId: string)
setUserProperties(properties: UserProperties)
```

### Hooks

- `useAnalytics()` - Main analytics hook
- `useScrollDepth(options)` - Scroll depth tracking
- `useUserBehavior(options)` - User behavior tracking

### Configuration

See `src/config/analytics.ts` for constants and configuration options.

---

## Additional Resources

- [Google Analytics 4 Documentation](https://developers.google.com/analytics/devguides/collection/ga4)
- [GDPR Compliance Guide](https://gdpr-info.eu/)
- [CCPA Compliance Guide](https://oag.ca.gov/privacy/ccpa)
- [Web Vitals](https://web.dev/vitals/)

---

Last Updated: November 2024
