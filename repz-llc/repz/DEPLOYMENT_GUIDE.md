# REPZ Platform - Complete Deployment Guide

## 🎯 **$800 Credit Utilization Strategy**

This guide will help you deploy the REPZ platform and activate all integrations using your $800 credit efficiently.

---

## **Table of Contents**

1. [Prerequisites](#prerequisites)
2. [Phase 1: Deploy to Vercel](#phase-1-deploy-to-vercel)
3. [Phase 2: Upgrade Supabase](#phase-2-upgrade-supabase)
4. [Phase 3: Configure Integrations](#phase-3-configure-integrations)
5. [Phase 4: Set Up Monitoring](#phase-4-set-up-monitoring)
6. [Phase 5: Configure Payments](#phase-5-configure-payments)
7. [Phase 6: Email & SMS Services](#phase-6-email--sms-services)
8. [Phase 7: Security & CDN](#phase-7-security--cdn)
9. [Phase 8: Final Testing](#phase-8-final-testing)

---

## **Prerequisites**

### Accounts You'll Need to Create:
- [ ] Vercel account (free, then upgrade to Pro)
- [ ] Supabase account (upgrade to Pro)
- [ ] Whoop Developer account
- [ ] Google Cloud Platform account
- [ ] Strava Developer account
- [ ] Apple Developer account ($99/year)
- [ ] Stripe account
- [ ] Sentry account
- [ ] SendGrid account
- [ ] Twilio account (optional)
- [ ] Cloudflare account

---

## **Phase 1: Deploy to Vercel** 💰 $20/month

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

### Step 2: Login to Vercel
```bash
vercel login
```

### Step 3: Link Your Project
```bash
cd /home/user/alawein-business/repz/REPZ/platform
vercel link
```

### Step 4: Configure Project Settings
```bash
# Set build command
vercel env add VITE_SUPABASE_URL production
vercel env add VITE_SUPABASE_ANON_KEY production
# Add more env vars as needed
```

### Step 5: Deploy to Production
```bash
npm run build:production
vercel --prod
```

### Step 6: Upgrade to Vercel Pro
1. Go to https://vercel.com/account/billing
2. Upgrade to Pro ($20/month)
3. Features unlocked:
   - Custom domains
   - Advanced analytics
   - Team collaboration
   - Better performance

**Cost: $60 for 3 months** ✅

---

## **Phase 2: Upgrade Supabase** 💰 $25/month

### Step 1: Access Your Supabase Project
1. Go to https://supabase.com/dashboard
2. Select your REPZ project

### Step 2: Upgrade to Pro
1. Click "Settings" → "Billing"
2. Upgrade to Pro plan ($25/month)
3. Benefits:
   - 8GB database
   - 50GB bandwidth
   - Daily backups
   - Point-in-time recovery
   - Priority support

### Step 3: Run Production Migrations
```bash
# Set production database URL
export DATABASE_URL="postgresql://..."

# Run migrations
npm run db:migrate

# Verify tables created
npm run db:seed:test
```

**Cost: $75 for 3 months** ✅

---

## **Phase 3: Configure Integrations**

### A. Whoop Integration 💰 $50/month (optional)

#### Step 1: Register Developer Account
1. Go to https://developer.whoop.com
2. Create developer account
3. Create new application

#### Step 2: Configure OAuth
1. **Application Name**: REPZ Platform
2. **Redirect URI**: `https://your-domain.com/integrations/whoop/callback`
3. **Scopes**:
   - `read:recovery`
   - `read:cycles`
   - `read:workout`
   - `read:sleep`
   - `read:profile`

#### Step 3: Get Credentials
```bash
# Add to Vercel environment
vercel env add VITE_WHOOP_CLIENT_ID production
vercel env add VITE_WHOOP_CLIENT_SECRET production
```

**Cost: $150 for 3 months** ✅

---

### B. Google Calendar Integration 💰 FREE (with limits)

#### Step 1: Create Google Cloud Project
1. Go to https://console.cloud.google.com
2. Create new project: "REPZ Platform"
3. Enable APIs:
   - Google Calendar API
   - Google OAuth2 API

#### Step 2: Configure OAuth Consent Screen
1. Go to "OAuth consent screen"
2. Select "External"
3. Fill in details:
   - **App name**: REPZ Platform
   - **User support email**: your-email
   - **Developer contact**: your-email

#### Step 3: Create OAuth Credentials
1. Go to "Credentials"
2. Create "OAuth 2.0 Client ID"
3. Application type: "Web application"
4. Authorized redirect URIs:
   - `https://your-domain.com/integrations/google-calendar/callback`
   - `http://localhost:8080/integrations/google-calendar/callback` (dev)

#### Step 4: Save Credentials
```bash
vercel env add VITE_GOOGLE_CLIENT_ID production
vercel env add VITE_GOOGLE_CLIENT_SECRET production
```

**Cost: $0 (free tier sufficient initially)** ✅

---

### C. Strava Integration 💰 FREE

#### Step 1: Register Application
1. Go to https://www.strava.com/settings/api
2. Create new application:
   - **Application Name**: REPZ Platform
   - **Category**: Training
   - **Website**: https://your-domain.com
   - **Authorization Callback Domain**: your-domain.com

#### Step 2: Configure Scopes
Required scopes:
- `read`
- `activity:read_all`
- `profile:read_all`

#### Step 3: Save Credentials
```bash
vercel env add VITE_STRAVA_CLIENT_ID production
vercel env add VITE_STRAVA_CLIENT_SECRET production
```

**Cost: $0 (free)** ✅

---

### D. Apple Developer Program 💰 $99/year

#### Step 1: Enroll in Apple Developer Program
1. Go to https://developer.apple.com/programs/enroll/
2. Enroll as Individual or Organization
3. Pay $99/year

#### Step 2: Enable HealthKit
1. Login to Apple Developer portal
2. Go to "Certificates, IDs & Profiles"
3. Enable HealthKit capability for your app

#### Step 3: Configure App IDs
1. Create App ID for REPZ
2. Enable capabilities:
   - HealthKit
   - Push Notifications
   - Background Modes

**Cost: $99 one-time** ✅

---

## **Phase 4: Set Up Monitoring** 💰 $26/month

### A. Sentry Error Tracking

#### Step 1: Create Sentry Account
1. Go to https://sentry.io/signup/
2. Create account
3. Create new project: "repz-platform"

#### Step 2: Get DSN
1. Go to Project Settings
2. Copy DSN
3. Add to environment:
```bash
vercel env add VITE_SENTRY_DSN production
```

#### Step 3: Install Sentry Integration
```bash
npm install --save @sentry/react @sentry/vite-plugin
```

#### Step 4: Configure Sentry
Already configured in the platform! Just add the DSN.

#### Step 5: Upgrade to Team Plan
- $26/month for 50K events
- Real-time error tracking
- Performance monitoring
- Release tracking

**Cost: $78 for 3 months** ✅

---

### B. Google Analytics (FREE)

#### Step 1: Create GA4 Property
1. Go to https://analytics.google.com
2. Create new GA4 property
3. Get Measurement ID (G-...)

#### Step 2: Add to Environment
```bash
vercel env add VITE_GA_MEASUREMENT_ID production
```

**Cost: $0 (free)** ✅

---

## **Phase 5: Configure Payments** 💰 Transaction fees only

### Stripe Setup

#### Step 1: Create Stripe Account
1. Go to https://dashboard.stripe.com/register
2. Complete business verification
3. Activate account

#### Step 2: Create Products
Create 4 products for each tier:
1. **Core Program** - $89/month
2. **Adaptive Engine** - $149/month
3. **Performance Suite** - $229/month
4. **Longevity Concierge** - $349/month

#### Step 3: Create Prices
For each product, create 4 billing periods:
- Monthly
- Quarterly (5% discount)
- Semi-annual (10% discount)
- Annual (15% discount)

#### Step 4: Get API Keys
```bash
# Add to Vercel
vercel env add VITE_STRIPE_PUBLIC_KEY production
vercel env add STRIPE_SECRET_KEY production

# Add all price IDs
vercel env add VITE_STRIPE_PRICE_CORE_MONTHLY production
# ... (add all 16 price IDs)
```

#### Step 5: Configure Webhooks
1. Go to Stripe Dashboard → Webhooks
2. Add endpoint: `https://your-domain.com/api/stripe-webhook`
3. Select events:
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.paid`
   - `invoice.payment_failed`

**Cost: 2.9% + $0.30 per transaction** ✅

---

## **Phase 6: Email & SMS Services**

### A. SendGrid 💰 $90/month

#### Step 1: Create SendGrid Account
1. Go to https://signup.sendgrid.com/
2. Complete verification
3. Upgrade to Pro ($89.95/month for 100K emails)

#### Step 2: Create API Key
1. Go to Settings → API Keys
2. Create new key with "Full Access"
3. Save securely

#### Step 3: Verify Domain
1. Go to Settings → Sender Authentication
2. Verify your domain
3. Configure DNS records

#### Step 4: Create Email Templates
Templates needed:
- Welcome email
- Password reset
- Subscription confirmation
- Workout reminders
- Referral invitations

#### Step 5: Add to Environment
```bash
vercel env add SENDGRID_API_KEY production
vercel env add SENDGRID_FROM_EMAIL production
```

**Cost: $90 for 1 month (then evaluate usage)** ✅

---

### B. Twilio (Optional) 💰 Pay-as-you-go

#### Step 1: Create Twilio Account
1. Go to https://www.twilio.com/try-twilio
2. Create account
3. Verify phone number

#### Step 2: Buy Phone Number
1. Buy a phone number (~$1/month)
2. Enable SMS

#### Step 3: Add Credits
- Add $50 for initial SMS credits
- ~$0.0075 per SMS

#### Step 4: Configure
```bash
vercel env add TWILIO_ACCOUNT_SID production
vercel env add TWILIO_AUTH_TOKEN production
vercel env add TWILIO_PHONE_NUMBER production
```

**Cost: $50 initial credits** ✅

---

## **Phase 7: Security & CDN** 💰 $20/month

### Cloudflare Pro

#### Step 1: Add Site to Cloudflare
1. Go to https://dash.cloudflare.com
2. Add your domain
3. Update nameservers at registrar

#### Step 2: Upgrade to Pro
1. $20/month for:
   - Advanced DDoS protection
   - Web Application Firewall
   - Image optimization
   - Polish (compression)
   - Advanced caching

#### Step 3: Configure Security
1. Enable "I'm Under Attack" mode if needed
2. Configure WAF rules
3. Set up rate limiting
4. Enable HTTPS/SSL

#### Step 4: Configure Caching
1. Set cache rules for static assets
2. Configure page rules
3. Enable Rocket Loader

**Cost: $60 for 3 months** ✅

---

## **Phase 8: Final Testing**

### Checklist

#### Infrastructure
- [ ] Vercel deployment successful
- [ ] Custom domain configured
- [ ] SSL certificate active
- [ ] Environment variables set
- [ ] Database migrations run

#### Integrations
- [ ] Whoop OAuth flow works
- [ ] Google Calendar sync works
- [ ] Strava activities sync
- [ ] Apple Health (test on iOS)

#### Payments
- [ ] Stripe test mode works
- [ ] All price IDs configured
- [ ] Webhooks receiving events
- [ ] Subscription flow complete

#### Monitoring
- [ ] Sentry capturing errors
- [ ] Google Analytics tracking
- [ ] Performance metrics visible

#### Communication
- [ ] SendGrid emails sending
- [ ] Email templates working
- [ ] SMS notifications (if enabled)

#### Security
- [ ] Cloudflare active
- [ ] Rate limiting configured
- [ ] Security headers present
- [ ] CORS properly configured

---

## **Cost Summary**

| Service | Monthly | 3-Month Total | Status |
|---------|---------|---------------|--------|
| Vercel Pro | $20 | $60 | ✅ |
| Supabase Pro | $25 | $75 | ✅ |
| Whoop API | $50 | $150 | Optional |
| Google Calendar | $0 | $0 | ✅ Free |
| Strava API | $0 | $0 | ✅ Free |
| Apple Developer | - | $99 | ✅ Annual |
| Sentry | $26 | $78 | ✅ |
| SendGrid | $90 | $90 | ✅ |
| Twilio | - | $50 | ✅ Credits |
| Cloudflare Pro | $20 | $60 | ✅ |
| **TOTAL** | **$231/mo** | **$662** | **Under budget!** |

**Remaining budget: $138** for contingencies! 🎉

---

## **Quick Start Commands**

```bash
# 1. Deploy to Vercel
cd /home/user/alawein-business/repz/REPZ/platform
vercel --prod

# 2. Set environment variables
vercel env pull .env.production

# 3. Run database migrations
npm run db:migrate

# 4. Test the deployment
curl https://your-domain.com/api/health

# 5. Monitor logs
vercel logs --prod
```

---

## **Next Steps After Deployment**

1. **Test All Integrations**
   - Connect your own accounts
   - Verify data sync
   - Test OAuth flows

2. **Set Up Monitoring Alerts**
   - Sentry alerts for errors
   - Uptime monitoring
   - Performance degradation alerts

3. **Marketing Setup**
   - Configure Google Analytics goals
   - Set up conversion tracking
   - Launch referral program

4. **User Onboarding**
   - Create welcome email sequence
   - Set up in-app tutorials
   - Prepare support documentation

5. **Performance Optimization**
   - Monitor page load times
   - Optimize images
   - Review bundle sizes

---

## **Support & Troubleshooting**

### Common Issues

**Issue: Deployment fails**
- Check build logs: `vercel logs`
- Verify all env vars are set
- Ensure dependencies installed

**Issue: Integration not working**
- Verify redirect URIs match exactly
- Check API credentials are correct
- Review OAuth scopes

**Issue: Database connection fails**
- Check Supabase connection string
- Verify RLS policies
- Check service role key

### Get Help
- Vercel Support: https://vercel.com/support
- Supabase Support: https://supabase.com/dashboard/support
- Documentation: See `/docs` folder
- GitHub Issues: Create an issue for bugs

---

## **Maintenance Schedule**

### Daily
- Monitor error rates (Sentry)
- Check uptime status
- Review user signups

### Weekly
- Review API usage and costs
- Check integration health
- Analyze user feedback

### Monthly
- Security audit
- Performance review
- Cost optimization
- Backup verification

---

**🎉 Congratulations! Your REPZ platform is production-ready!**

*Last updated: January 18, 2025*
