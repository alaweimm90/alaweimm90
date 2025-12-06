# ✅ REPZ Analytics Testing - Quick Start Guide

Your Google Analytics and Tag Manager setup is now configured with your actual production IDs:

## 🎯 Your Configuration

- **GTM Container:** `GTM-K3996XDS` (REPZ Container)
- **GA4 Measurement ID:** `G-XL2VMRCZS2` (REPZ stream)
- **Account:** REPZ (ID: 364739085)
- **Stream ID:** 12013494336

## 🧪 Quick Test Steps

### 1. Browser Console Testing

Open your site and run these commands in the browser console:

```javascript
// 1. Check if everything is loaded correctly
GTMHelpers.runDiagnostic()

// Expected output:
// 🔍 GTM Diagnostic Starting...
// 📋 Configuration: {containerId: "GTM-K3996XDS", gaId: "G-XL2VMRCZS2", debugMode: true}
// ✅ Diagnostic test event sent

// 2. Test REPZ-specific events
GTMHelpers.trackTierSelection('performance', 'monthly', 229)
GTMHelpers.trackBookingAttempt('personalTrainingMonthly', 'home')
GTMHelpers.trackFeatureUsage('ai_assistant', 'performance')
```

### 2. GTM Preview Mode Testing

1. **Go to GTM:** https://tagmanager.google.com/
2. **Select:** REPZ Container (GTM-K3996XDS)
3. **Click:** Preview button
4. **Enter URL:** Your site URL (localhost:8080 or production)
5. **Navigate:** The opened tab to your site

**You should see:**
- GTM Preview panel connected at bottom
- Events firing as you interact with the site
- Custom REPZ events appearing in real-time

### 3. GA4 Verification

1. **Real-time:** Go to [analytics.google.com](https://analytics.google.com) → Reports → Real-time
2. **DebugView:** Go to Configure → DebugView
3. **Check:** Events appear within 30 seconds of interaction

## 🎯 Test These REPZ Features

### Tier Selection Testing
1. Navigate to pricing page
2. Click different tier cards
3. Check GTM Preview for `tier_selected` events

### Booking Testing
1. Go to in-person training page
2. Click "Book Session" buttons
3. Verify `booking_attempt` events fire

### Dashboard Testing
1. Login to dashboard
2. Use AI features
3. Log workouts/nutrition
4. Check for `feature_usage`, `workout_logged`, `nutrition_logged` events

## 🔧 Your GTM Container Should Have

Based on your configuration, ensure your GTM container has these tags:

✅ **Google Analytics GA4 Event** - Firing on All Pages  
✅ **Google Tag** - Firing on Initialization  
✅ **Google Tag 1** - Firing on Initialization  

And these built-in variables:
✅ **Event** - Custom Event  
✅ **Page Hostname** - URL  
✅ **Page Path** - URL  
✅ **Page URL** - URL  
✅ **Referrer** - HTTP Referrer  

## 🚨 Troubleshooting

If events aren't firing:

```javascript
// Check basic setup
console.log('GTM loaded:', !!window.google_tag_manager)
console.log('DataLayer:', window.dataLayer?.length || 0)
console.log('Container ID:', GTMHelpers.GTM_CONFIG.containerId)

// Test manual event
window.dataLayer = window.dataLayer || []
window.dataLayer.push({
  event: 'test_event_manual',
  test_parameter: 'working'
})
```

## 📱 Next Steps

1. **Test on localhost** first
2. **Deploy to staging** and test
3. **Deploy to production** (repzcoach.com)
4. **Monitor in GA4** for 24-48 hours

Your setup is ready to go! The tags are installed in `index.html` and all configurations use your real production IDs.

---

**Container:** GTM-K3996XDS | **GA4:** G-XL2VMRCZS2 | **Stream:** 12013494336