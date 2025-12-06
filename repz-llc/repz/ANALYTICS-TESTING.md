# Google Analytics & Tag Manager Testing Guide

This guide provides step-by-step instructions for testing Google Tag Manager (GTM) and Google Analytics 4 (GA4) implementation on the REPZ platform.

## 📋 Prerequisites

1. **Environment Variables Configured:**
   ```bash
   GOOGLE_TAG_MANAGER_ID=GTM-K3996XDS  # REPZ Production GTM Container
   GOOGLE_ANALYTICS_ID=G-XL2VMRCZS2    # REPZ Production GA4 Measurement ID
   ```

2. **GTM Container Setup:**
   - GA4 Configuration tag configured with your measurement ID
   - Custom events configured for REPZ-specific tracking
   - Triggers set up for user interactions

## 🔧 Testing Setup

### 1. Enable GTM Preview Mode

1. **Go to Google Tag Manager:**
   - Navigate to [https://tagmanager.google.com/](https://tagmanager.google.com/)
   - Select your REPZ container (GTM-K3996XDS)
   - Click **"Preview"** button

2. **Connect to Your Site:**
   - In preview mode, enter your site URL:
     - Development: `http://localhost:8080`
     - Staging: Your staging URL
     - Production: `https://repzcoach.pro`
   - Click **"Connect"**
   - Navigate to your site in the opened tab

3. **Verify Connection:**
   - You should see the GTM Preview panel at the bottom
   - Panel shows "Connected" status
   - Tag firing information appears as you navigate

### 2. Browser Console Testing

Open your browser's developer console and run:

```javascript
// Run comprehensive diagnostic
GTMHelpers.runDiagnostic()

// Check setup validity
GTMHelpers.validateSetup()

// Enable debug logging
GTMHelpers.debugGA4()
```

**Expected Output:**
```
🔍 GTM Diagnostic Starting...
📋 Configuration: {containerId: "GTM-K3996XDS", gaId: "G-XL2VMRCZS2", debugMode: true}
📊 DataLayer exists: true
📊 DataLayer length: 5
🏷️ GTM Container loaded: true
📈 GA4 gtag loaded: true
✅ Diagnostic test event sent
```

## 🎯 Event Testing

### 3. Test REPZ-Specific Events

Execute these commands in the browser console to test event tracking:

```javascript
// Test tier selection
GTMHelpers.trackTierSelection('performance', 'monthly', 229)

// Test booking attempt
GTMHelpers.trackBookingAttempt('personalTrainingMonthly', 'home')

// Test feature usage
GTMHelpers.trackFeatureUsage('ai_assistant', 'performance')

// Test workout logging
GTMHelpers.trackWorkoutLog('strength_training', 45)

// Test nutrition logging
GTMHelpers.trackNutritionLog('breakfast', 520)

// Test AI interaction
GTMHelpers.trackAIInteraction('form_analysis', 'feedback_requested')
```

### 4. Interactive Testing

Navigate through the application and perform these actions:

1. **Subscription Flow:**
   - Visit pricing page
   - Click on different tiers
   - Change billing cycles
   - Start checkout process

2. **Booking Flow:**
   - Navigate to in-person training
   - Click booking buttons
   - Fill out Calendly forms

3. **Dashboard Interactions:**
   - Log workouts
   - Track nutrition
   - Use AI features
   - Submit forms

4. **Feature Usage:**
   - Access tier-specific features
   - Use analytics dashboards
   - Interact with coaching tools

## 📊 Verification Steps

### 5. GTM Preview Verification

In the GTM Preview panel, verify these events fire:

**Page Views:**
- `page_view` - Every page navigation
- `gtm.dom` - DOM ready
- `gtm.load` - Window loaded

**Custom Events:**
- `tier_selected` - Tier selection events
- `booking_attempt` - Booking interactions
- `feature_usage` - Feature access events
- `form_submission` - Form completions
- `workout_logged` - Fitness tracking
- `nutrition_logged` - Nutrition tracking
- `ai_interaction` - AI feature usage

**Event Parameters:**
- Check that custom parameters are captured correctly
- Verify user_tier, feature_name, booking_type etc.

### 6. GA4 Real-time Verification

1. **Open GA4 Real-time Reports:**
   - Go to [analytics.google.com](https://analytics.google.com)
   - Navigate to **Reports → Real-time**
   - You should see active users and events

2. **Check Event Stream:**
   - Events should appear within 30 seconds
   - Verify event names match expectations
   - Check event parameters are populated

### 7. GA4 DebugView Verification

1. **Enable Debug Mode:**
   - In GA4, go to **Configure → DebugView**
   - You should see your debug device listed
   - Events appear in real-time with full parameter details

2. **Verify Event Details:**
   - Click on events to see parameters
   - Check custom dimensions are captured
   - Verify enhanced ecommerce data (for subscriptions)

## 🐛 Troubleshooting

### Common Issues and Solutions

#### GTM Not Loading
```javascript
// Check if GTM script is present
console.log('GTM script loaded:', !!window.google_tag_manager);

// Check container ID
console.log('Container ID:', GTMHelpers.GTM_CONFIG.containerId);

// Manually initialize if needed
GTMHelpers.init('GTM-K3996XDS');
```

#### Events Not Firing
```javascript
// Check dataLayer
console.log('DataLayer:', window.dataLayer);

// Test manual event
window.dataLayer = window.dataLayer || [];
window.dataLayer.push({
  event: 'test_event',
  test_parameter: 'test_value'
});
```

#### GA4 Not Receiving Events
```javascript
// Check GA4 configuration
console.log('GA4 ID:', GTMHelpers.GTM_CONFIG.gaId);

// Test direct GA4 event
if (window.gtag) {
  window.gtag('event', 'test_action', {
    event_category: 'test',
    event_label: 'manual_test'
  });
}
```

#### Preview Mode Connection Issues
1. **Clear browser cache and cookies**
2. **Disable ad blockers**
3. **Use incognito/private browsing mode**
4. **Check for JavaScript errors in console**
5. **Verify GTM container is published**

## 📈 Analytics Dashboard Setup

### 8. GA4 Custom Reports

Create custom reports for REPZ-specific metrics:

1. **Subscription Funnel:**
   - Event: `tier_selected`
   - Parameters: `tier_name`, `billing_cycle`, `price_amount`

2. **Booking Analysis:**
   - Event: `booking_attempt`
   - Parameters: `booking_type`, `location`

3. **Feature Usage:**
   - Event: `feature_usage`
   - Parameters: `feature_name`, `user_tier`

4. **Fitness Tracking:**
   - Events: `workout_logged`, `nutrition_logged`
   - Parameters: `workout_type`, `duration_minutes`, `calories`

### 9. GTM Variables and Triggers

Set up these variables in GTM:

```javascript
// User Tier Variable
{{User Tier}} = dataLayer.user_tier || 'unknown'

// Page Category Variable
{{Page Category}} = dataLayer.page_category || 'general'

// Feature Access Variable  
{{Feature Access}} = dataLayer.feature_access || 'basic'
```

Create triggers for:
- Page views (all pages)
- Tier selection events
- Booking interactions
- Feature usage events
- Form submissions

## ✅ Validation Checklist

### Pre-Launch Validation

- [ ] GTM container loads successfully
- [ ] GA4 configuration tag fires on all pages
- [ ] Custom events fire correctly
- [ ] Event parameters are captured accurately
- [ ] User identification works properly
- [ ] Enhanced ecommerce tracking functions
- [ ] Real-time data appears in GA4
- [ ] DebugView shows detailed event data
- [ ] No JavaScript errors in console
- [ ] Preview mode connects successfully

### Production Monitoring

- [ ] Set up GA4 alerts for:
  - Sudden traffic drops
  - Conversion rate changes
  - Error event spikes
  - Missing event data

- [ ] Monitor GTM for:
  - Tag firing errors
  - High bounce rates from tracking issues
  - Performance impact

## 🎯 Advanced Testing

### 10. Automated Testing

Add GTM/GA4 validation to your test suite:

```typescript
// Example E2E test
test('GTM events fire correctly', async ({ page }) => {
  // Track dataLayer events
  const dataLayerEvents: any[] = [];
  
  await page.evaluateOnNewDocument(() => {
    window.dataLayer = window.dataLayer || [];
    const originalPush = window.dataLayer.push;
    window.dataLayer.push = function(...args) {
      // Store events for testing
      (window as any).testDataLayerEvents = (window as any).testDataLayerEvents || [];
      (window as any).testDataLayerEvents.push(...args);
      return originalPush.apply(this, args);
    };
  });

  await page.goto('/pricing');
  await page.click('[data-testid="tier-performance-select"]');
  
  const events = await page.evaluate(() => (window as any).testDataLayerEvents || []);
  
  expect(events.some(e => e.event === 'tier_selected')).toBe(true);
  expect(events.find(e => e.event === 'tier_selected')?.tier_name).toBe('performance');
});
```

### 11. Performance Testing

Monitor GTM/GA4 impact on site performance:

```javascript
// Check GTM load time
performance.mark('gtm-start');
// ... GTM initialization
performance.mark('gtm-end');
performance.measure('gtm-load-time', 'gtm-start', 'gtm-end');

// Get measurement
const gtmloadTime = performance.getEntriesByName('gtm-load-time')[0];
console.log('GTM load time:', gtmloadTime.duration, 'ms');
```

## 📞 Support and Resources

### Documentation
- [GTM Developer Guide](https://developers.google.com/tag-manager)
- [GA4 Implementation Guide](https://developers.google.com/analytics/devguides/collection/ga4)
- [Enhanced Ecommerce Setup](https://developers.google.com/analytics/devguides/collection/ga4/ecommerce)

### Debugging Tools
- [GA4 DebugView](https://support.google.com/analytics/answer/7201382)
- [GTM Preview Mode](https://support.google.com/tagmanager/answer/6107056)
- [Google Analytics Debugger Extension](https://chrome.google.com/webstore/detail/google-analytics-debugger)

---

**Last Updated:** January 2025  
**Maintainer:** REPZ Development Team  
**Status:** Production Ready ✅