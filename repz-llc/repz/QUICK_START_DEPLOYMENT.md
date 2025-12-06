# 🚀 REPZ Platform - Quick Start Deployment

## **Get Your Platform Live in 30 Minutes!**

This guide gets REPZ deployed to production FAST. Full details in [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md).

---

## **Prerequisites (5 minutes)**

1. **Install Vercel CLI**
```bash
npm install -g vercel
```

2. **Login to Vercel**
```bash
vercel login
```

3. **Have these accounts ready:**
   - Supabase account
   - Stripe account (for payments)
   - OpenAI API key

---

## **Deploy in 3 Commands** ⚡

### **Step 1: Configure Environment** (10 minutes)
```bash
# Interactive setup wizard
./scripts/setup-integrations.sh
```

This will ask for:
- Supabase URL and keys
- Stripe API keys
- Optional: Whoop, Google, Strava credentials
- Optional: SendGrid, Sentry

### **Step 2: Deploy to Production** (15 minutes)
```bash
# Automated deployment script
./scripts/deploy-production.sh
```

This will:
- Run type checks
- Build production version
- Deploy to Vercel
- Give you a live URL!

### **Step 3: Run Database Migrations** (5 minutes)
```bash
# Connect to production database
npm run db:migrate
```

---

## **Done! 🎉**

Your REPZ platform is now live!

**Access your platform:**
- Production URL: `https://your-project.vercel.app`
- Custom domain: Configure in Vercel dashboard

---

## **Next Steps**

### **1. Upgrade Services** (Do this within 24 hours)

#### Vercel Pro ($20/month)
```bash
# Go to: https://vercel.com/account/billing
# Click "Upgrade to Pro"
```

#### Supabase Pro ($25/month)
```bash
# Go to: https://supabase.com/dashboard
# Settings → Billing → Upgrade to Pro
```

### **2. Test All Features**

```bash
# Test the platform
curl https://your-project.vercel.app/api/health

# Check database connection
npm run db:seed:test

# View logs
vercel logs --prod
```

### **3. Configure Integrations** (Optional but recommended)

**Whoop** (~$50/month)
1. Register at https://developer.whoop.com
2. Create app, get credentials
3. Add to Vercel: `vercel env add VITE_WHOOP_CLIENT_ID production`

**Google Calendar** (FREE)
1. Create project at https://console.cloud.google.com
2. Enable Calendar API
3. Create OAuth credentials

**Strava** (FREE)
1. Register app at https://www.strava.com/settings/api
2. Get client ID and secret

**Apple Developer** ($99/year)
1. Enroll at https://developer.apple.com/programs/
2. Enable HealthKit capability

### **4. Set Up Monitoring**

**Sentry** ($26/month)
```bash
# Sign up: https://sentry.io
# Get DSN and add to environment
vercel env add VITE_SENTRY_DSN production
```

**Google Analytics** (FREE)
```bash
# Create GA4 property
# Add measurement ID to environment
vercel env add VITE_GA_MEASUREMENT_ID production
```

### **5. Configure Email**

**SendGrid** ($90/month for 100K emails)
```bash
# Sign up: https://signup.sendgrid.com
# Verify domain
# Create API key
vercel env add SENDGRID_API_KEY production
```

---

## **Cost Breakdown**

### **Required (Month 1)**
| Service | Cost | What You Get |
|---------|------|--------------|
| Vercel Pro | $20 | Hosting, custom domain, analytics |
| Supabase Pro | $25 | 8GB database, 50GB bandwidth |
| OpenAI API | ~$20 | AI features (usage-based) |
| **Total Required** | **$65** | **Core platform live** |

### **Recommended (Month 1)**
| Service | Cost | What You Get |
|---------|------|--------------|
| Above Required | $65 | Core platform |
| Apple Developer | $99/year | iOS HealthKit integration |
| Sentry | $26 | Error tracking |
| SendGrid | $90 | Email service |
| **Total Recommended** | **$189** | **Full-featured platform** |

### **Full Featured (Month 1)**
| Service | Cost | What You Get |
|---------|------|--------------|
| Above | $189 | Full platform |
| Whoop API | $50 | Biometric tracking |
| Cloudflare Pro | $20 | Advanced security/CDN |
| Twilio | $50 | SMS notifications (credits) |
| **Total Full** | **$309** | **Everything activated!** |

**3-Month Total: ~$700** (well under your $800 budget!)

---

## **Troubleshooting**

### **Deployment fails?**
```bash
# Check build logs
vercel logs

# Verify environment variables
vercel env ls

# Re-run build locally
npm run build:production
```

### **Database connection fails?**
```bash
# Check connection string
echo $VITE_SUPABASE_URL

# Test connection
npm run db:migrate:test

# Verify RLS policies
# Check: https://supabase.com/dashboard/project/_/auth/policies
```

### **Integration not working?**
```bash
# Verify OAuth redirect URIs match
# Check API credentials are correct
# Review scopes requested

# Test locally first
npm run dev
```

---

## **Quick Reference Commands**

```bash
# Deploy to production
./scripts/deploy-production.sh

# Configure integrations
./scripts/setup-integrations.sh

# Add environment variable
vercel env add VAR_NAME production

# View production logs
vercel logs --prod

# Run database migrations
npm run db:migrate

# Seed test data
npm run db:seed

# Type check
npm run type-check

# Build for production
npm run build:production
```

---

## **Important URLs**

### **Your Platforms**
- **Production**: https://your-project.vercel.app
- **Supabase Dashboard**: https://supabase.com/dashboard
- **Vercel Dashboard**: https://vercel.com/dashboard

### **API Registrations**
- **Whoop**: https://developer.whoop.com
- **Google Cloud**: https://console.cloud.google.com
- **Strava**: https://www.strava.com/settings/api
- **Apple**: https://developer.apple.com
- **Stripe**: https://dashboard.stripe.com
- **Sentry**: https://sentry.io
- **SendGrid**: https://app.sendgrid.com

### **Documentation**
- **Full Deployment Guide**: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- **Integration Docs**: [docs/INTEGRATIONS.md](./docs/INTEGRATIONS.md)
- **GDPR Compliance**: [docs/GDPR_COMPLIANCE.md](./docs/GDPR_COMPLIANCE.md)

---

## **Support**

- **Questions?** Check [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- **Issues?** Create a GitHub issue
- **Need help?** Contact support@repz-platform.com

---

**🎉 Happy Deploying!**

*Your platform will be live in 30 minutes or less!*

---

*Last updated: January 18, 2025*
