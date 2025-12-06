# REPZ Coach Pro - Production Configuration Guide

This document contains all production configuration data for the REPZ Coach Pro platform. This information is critical for deployment and integration setup.

## 📋 Table of Contents
- [Stripe Product & Price IDs](#stripe-product--price-ids)
- [Calendly Integration URLs](#calendly-integration-urls)
- [Google Analytics & Tag Manager](#google-analytics--tag-manager)
- [Environment Variables Setup](#environment-variables-setup)
- [Integration Instructions](#integration-instructions)

## 💳 Stripe Product & Price IDs

### Core Program ($89/month base)
**Product ID:** `prod_RJfv0P4A7vWF5c`

**Price IDs:**
- Monthly: `price_1QkCJpAb5lqTiGnqjjW6WSnF`
- Quarterly: `price_1QkCJpAb5lqTiGnqC2pNWOKr`
- Semi-Annual: `price_1QkCJpAb5lqTiGnq7aVgWnBo`
- Annual: `price_1QkCJpAb5lqTiGnqVKNnFPJD`

### Adaptive Engine ($149/month base)
**Product ID:** `prod_RJfw8mh3OG6z9Q`

**Price IDs:**
- Monthly: `price_1QkCKZAb5lqTiGnqTSQgRrBh`
- Quarterly: `price_1QkCKZAb5lqTiGnqZ8z9vxCC`
- Semi-Annual: `price_1QkCKZAb5lqTiGnqJ5TnUHVP`
- Annual: `price_1QkCKZAb5lqTiGnqUjLQcZdX`

### Performance Suite ($229/month base)
**Product ID:** `prod_RJfwvdgYZR6gHl`

**Price IDs:**
- Monthly: `price_1QkCLBAb5lqTiGnqLgZWL8E0`
- Quarterly: `price_1QkCLBAb5lqTiGnqGfcT2Xrf`
- Semi-Annual: `price_1QkCLBAb5lqTiGnqBJhzgkQu`
- Annual: `price_1QkCLBAb5lqTiGnq7j9pT1Uu`

### Longevity Concierge ($349/month base)
**Product ID:** `prod_RJfxQnQmOLQWiR`

**Price IDs:**
- Monthly: `price_1QkCLnAb5lqTiGnqzfAKPyim`
- Quarterly: `price_1QkCLnAb5lqTiGnq9VQJMnYx`
- Semi-Annual: `price_1QkCLnAb5lqTiGnqNhZ8NfPU`
- Annual: `price_1QkCLnAb5lqTiGnqv6WJpYKE`

## 📅 Calendly Integration URLs

### In-Person Training Sessions
- **Monthly Training:** `https://calendly.com/repzmeshacoach/personal-training-monthly`
- **Semi-Weekly Training:** `https://calendly.com/repzmeshacoach/personal-training-semi-weekly`

### Consultation & Assessment Calls
- **Initial Consultation:** `https://calendly.com/repzmeshacoach/consultation-call`
- **Progress Review:** `https://calendly.com/repzmeshacoach/progress-review`
- **Strategy Session:** `https://calendly.com/repzmeshacoach/strategy-session`

## 📊 Google Analytics & Tag Manager

### Google Tag Manager (GTM)
**Container ID:** `GTM-K3996XDS` *(REPZ Production)*
**Account:** REPZ
**Container:** REPZ Container

**GTM Installation Code:**
```html
<!-- Google Tag Manager -->
<script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-K3996XDS');</script>
<!-- End Google Tag Manager -->

<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-K3996XDS"
height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
<!-- End Google Tag Manager (noscript) -->
```

### Google Analytics 4 (GA4)
**Measurement ID:** `G-XL2VMRCZS2` *(REPZ Production)*
**Account:** REPZ (ID: 364739085)  
**Stream:** REPZ stream (ID: 12013494336)
**Website URL:** https://www.repzcoach.com

**GA4 Installation Code:**
```html
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XL2VMRCZS2"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XL2VMRCZS2');
</script>
```

## ⚙️ Environment Variables Setup

### Update your `.env.local` file with these production Stripe IDs:

```bash
# Stripe Configuration - Production
STRIPE_SECRET_KEY=sk_live_xxxxxxxxxxxxxxxxxxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxxxxxxxxxx

# Core Program ($89/month)
STRIPE_PRICE_CORE_MONTHLY_PROD=price_1QkCJpAb5lqTiGnqjjW6WSnF
STRIPE_PRICE_CORE_QUARTERLY_PROD=price_1QkCJpAb5lqTiGnqC2pNWOKr
STRIPE_PRICE_CORE_SEMIANNUAL_PROD=price_1QkCJpAb5lqTiGnq7aVgWnBo
STRIPE_PRICE_CORE_ANNUAL_PROD=price_1QkCJpAb5lqTiGnqVKNnFPJD

# Adaptive Engine ($149/month)
STRIPE_PRICE_ADAPTIVE_MONTHLY_PROD=price_1QkCKZAb5lqTiGnqTSQgRrBh
STRIPE_PRICE_ADAPTIVE_QUARTERLY_PROD=price_1QkCKZAb5lqTiGnqZ8z9vxCC
STRIPE_PRICE_ADAPTIVE_SEMIANNUAL_PROD=price_1QkCKZAb5lqTiGnqJ5TnUHVP
STRIPE_PRICE_ADAPTIVE_ANNUAL_PROD=price_1QkCKZAb5lqTiGnqUjLQcZdX

# Performance Suite ($229/month)
STRIPE_PRICE_PERFORMANCE_MONTHLY_PROD=price_1QkCLBAb5lqTiGnqLgZWL8E0
STRIPE_PRICE_PERFORMANCE_QUARTERLY_PROD=price_1QkCLBAb5lqTiGnqGfcT2Xrf
STRIPE_PRICE_PERFORMANCE_SEMIANNUAL_PROD=price_1QkCLBAb5lqTiGnqBJhzgkQu
STRIPE_PRICE_PERFORMANCE_ANNUAL_PROD=price_1QkCLBAb5lqTiGnq7j9pT1Uu

# Longevity Concierge ($349/month)
STRIPE_PRICE_LONGEVITY_MONTHLY_PROD=price_1QkCLnAb5lqTiGnqzfAKPyim
STRIPE_PRICE_LONGEVITY_QUARTERLY_PROD=price_1QkCLnAb5lqTiGnq9VQJMnYx
STRIPE_PRICE_LONGEVITY_SEMIANNUAL_PROD=price_1QkCLnAb5lqTiGnqNhZ8NfPU
STRIPE_PRICE_LONGEVITY_ANNUAL_PROD=price_1QkCLnAb5lqTiGnqv6WJpYKE

# Calendly Integration
CALENDLY_PERSONAL_TRAINING_MONTHLY=https://calendly.com/repzmeshacoach/personal-training-monthly
CALENDLY_PERSONAL_TRAINING_SEMI_WEEKLY=https://calendly.com/repzmeshacoach/personal-training-semi-weekly
CALENDLY_CONSULTATION_CALL=https://calendly.com/repzmeshacoach/consultation-call
CALENDLY_PROGRESS_REVIEW=https://calendly.com/repzmeshacoach/progress-review
CALENDLY_STRATEGY_SESSION=https://calendly.com/repzmeshacoach/strategy-session

# Google Analytics & Tag Manager - REPZ Production
GOOGLE_ANALYTICS_ID=G-XL2VMRCZS2
GOOGLE_TAG_MANAGER_ID=GTM-K3996XDS
```

## 🧪 Testing Your Setup

### Analytics Testing (Google Tag Manager & GA4)

Once your environment variables are configured, test your analytics implementation:

```javascript
// Open browser console and run:
GTMHelpers.runDiagnostic()
GTMHelpers.validateSetup()
GTMHelpers.debugGA4()

// Test REPZ-specific events:
GTMHelpers.trackTierSelection('performance', 'monthly', 229)
GTMHelpers.trackBookingAttempt('personalTrainingMonthly', 'home')
GTMHelpers.trackFeatureUsage('ai_assistant', 'performance')
```

**Step-by-step testing guide:** See `ANALYTICS-TESTING.md` for complete instructions.

### Calendly Testing

Test booking buttons by:
1. Navigate to in-person training page
2. Click any "Book Session" button
3. Verify correct Calendly URL opens with your repzmeshacoach links
4. Check that user tier and preferences are prefilled

## 🔧 Integration Instructions

### For Developers & Claude Code:

#### 1. Stripe Integration Setup
```typescript
// Use these environment variables in your Stripe integration
const stripeConfig = {
  core: {
    monthly: process.env.STRIPE_PRICE_CORE_MONTHLY_PROD,
    quarterly: process.env.STRIPE_PRICE_CORE_QUARTERLY_PROD,
    semiannual: process.env.STRIPE_PRICE_CORE_SEMIANNUAL_PROD,
    annual: process.env.STRIPE_PRICE_CORE_ANNUAL_PROD
  },
  adaptive: {
    monthly: process.env.STRIPE_PRICE_ADAPTIVE_MONTHLY_PROD,
    quarterly: process.env.STRIPE_PRICE_ADAPTIVE_QUARTERLY_PROD,
    semiannual: process.env.STRIPE_PRICE_ADAPTIVE_SEMIANNUAL_PROD,
    annual: process.env.STRIPE_PRICE_ADAPTIVE_ANNUAL_PROD
  },
  performance: {
    monthly: process.env.STRIPE_PRICE_PERFORMANCE_MONTHLY_PROD,
    quarterly: process.env.STRIPE_PRICE_PERFORMANCE_QUARTERLY_PROD,
    semiannual: process.env.STRIPE_PRICE_PERFORMANCE_SEMIANNUAL_PROD,
    annual: process.env.STRIPE_PRICE_PERFORMANCE_ANNUAL_PROD
  },
  longevity: {
    monthly: process.env.STRIPE_PRICE_LONGEVITY_MONTHLY_PROD,
    quarterly: process.env.STRIPE_PRICE_LONGEVITY_QUARTERLY_PROD,
    semiannual: process.env.STRIPE_PRICE_LONGEVITY_SEMIANNUAL_PROD,
    annual: process.env.STRIPE_PRICE_LONGEVITY_ANNUAL_PROD
  }
};
```

#### 2. Calendly Integration
```typescript
// Calendly booking URLs by service type
const calendlyUrls = {
  personalTraining: {
    monthly: process.env.CALENDLY_PERSONAL_TRAINING_MONTHLY,
    semiWeekly: process.env.CALENDLY_PERSONAL_TRAINING_SEMI_WEEKLY
  },
  consultations: {
    initial: process.env.CALENDLY_CONSULTATION_CALL,
    progressReview: process.env.CALENDLY_PROGRESS_REVIEW,
    strategy: process.env.CALENDLY_STRATEGY_SESSION
  }
};
```

#### 3. Analytics Integration
Add the Google Tag Manager and Google Analytics scripts to your `index.html` or main layout component using the codes provided above.

### Important Notes:
1. **Security**: Never commit actual API keys or secrets to version control
2. **Environment**: Use production IDs only in production environment
3. **Testing**: Keep test mode price IDs for development/staging environments
4. **Validation**: Always validate environment variables are loaded correctly
5. **Monitoring**: Set up alerts for failed payment processing or integration issues

## 🚨 Critical Reminders

- **Stripe Webhooks**: Configure webhook endpoints for subscription lifecycle events
- **Calendly Webhooks**: Set up webhooks for booking confirmations and cancellations
- **Analytics Events**: Configure custom events for subscription conversions and user interactions
- **Error Handling**: Implement proper error handling for all payment and booking flows
- **Rate Limiting**: Implement rate limiting for API calls to external services

---

**Last Updated:** January 2025  
**Maintainer:** REPZ Development Team  
**Status:** Production Ready ✅

*This configuration file should be updated whenever new products, prices, or integration endpoints are created.*