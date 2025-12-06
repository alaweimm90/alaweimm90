# 🔥 REPZ $380 Deployment Checklist

**Goal:** Launch REPZ in 60 minutes with $380 budget

**Start Time:** __________ | **Target End:** __________ (+60 min)

---

## ✅ Phase 1: Core Infrastructure (15 minutes)

### 1.1 Vercel Pro - $60 (3 months)
- [ ] Open: https://vercel.com/signup
- [ ] Sign up with GitHub
- [ ] Import repository: `alaweimm90-business/alaweimm90-business`
- [ ] Select project: `repz/REPZ/platform`
- [ ] **Upgrade to Pro:**
  - [ ] Go to Settings → Billing
  - [ ] Select Pro plan ($20/month)
  - [ ] Add payment method
  - [ ] Confirm $60 for 3 months
- [ ] **Note Project URL:** __________________.vercel.app
- [ ] **Cost:** $60 ✅

### 1.2 Supabase Pro - $75 (3 months)
- [ ] Open: https://supabase.com/dashboard
- [ ] Find your REPZ project
- [ ] Go to Settings → Billing
- [ ] **Upgrade to Pro:**
  - [ ] Select Pro plan ($25/month)
  - [ ] Add payment method
  - [ ] Prepay 3 months: $75
- [ ] **Copy credentials:**
  - [ ] Project URL: __________________
  - [ ] Anon Key: __________________
  - [ ] Service Role Key: __________________
- [ ] **Cost:** $75 ✅

### 1.3 Domain Name - $15
- [ ] Open: https://www.namecheap.com/domains/
- [ ] Search: `repzcoach` or `repzfitness` or `getrepz`
- [ ] Select domain: __________________.com
- [ ] Add to cart
- [ ] Enable WhoisGuard (free)
- [ ] Complete purchase: $15
- [ ] **Note nameservers for later**
- [ ] **Cost:** $15 ✅

**Phase 1 Total: $150 | Time: _____ minutes**

---

## ✅ Phase 2: Revenue Enablers (15 minutes)

### 2.1 Stripe - $0 (FREE setup)
- [ ] Open: https://dashboard.stripe.com/register
- [ ] Sign up
- [ ] Complete business verification
- [ ] **Create Products:**
  - [ ] REPZ Basic: $299.00 (one-time payment)
  - [ ] REPZ Premium: $599.00 (one-time payment)
  - [ ] REPZ Concierge: $1,499.00 (one-time payment)
- [ ] **Get API Keys:**
  - [ ] Go to Developers → API keys
  - [ ] Copy Publishable key: pk_live_________________
  - [ ] Copy Secret key: sk_live_________________
- [ ] **Set up Webhook:**
  - [ ] Go to Developers → Webhooks
  - [ ] Add endpoint: https://your-project.vercel.app/api/stripe-webhook
  - [ ] Events: `checkout.session.completed`, `payment_intent.succeeded`
  - [ ] Copy Signing secret: whsec_________________
- [ ] **Cost:** $0 ✅

### 2.2 SendGrid Essentials - $60 (3 months)
- [ ] Open: https://signup.sendgrid.com/
- [ ] Sign up
- [ ] Select Essentials plan ($20/month)
- [ ] Add payment method
- [ ] Prepay 3 months: $60
- [ ] **Verify Domain:**
  - [ ] Go to Settings → Sender Authentication
  - [ ] Click "Authenticate Your Domain"
  - [ ] Enter your domain from 1.3
  - [ ] Add DNS records to Namecheap
- [ ] **Create API Key:**
  - [ ] Go to Settings → API Keys
  - [ ] Create API Key: SG.________________
  - [ ] Set permissions: Full Access
- [ ] **Cost:** $60 ✅

**Phase 2 Total: $60 | Time: _____ minutes | Running Total: $210**

---

## ✅ Phase 3: Professional Setup (15 minutes)

### 3.1 Google Workspace - $36 (2 users, 3 months)
- [ ] Open: https://workspace.google.com/
- [ ] Start free trial
- [ ] Enter domain from 1.3
- [ ] **Create users:**
  - [ ] coach@yourdomain.com
  - [ ] support@yourdomain.com
- [ ] **Upgrade to paid:**
  - [ ] Business Starter: $6/user/month
  - [ ] 2 users × 3 months = $36
- [ ] **Verify domain:**
  - [ ] Add TXT record to Namecheap DNS
  - [ ] Add MX records for email
- [ ] **Cost:** $36 ✅

### 3.2 Sentry Team Plan - $78 (3 months)
- [ ] Open: https://sentry.io/signup/
- [ ] Sign up
- [ ] Select Team plan ($26/month)
- [ ] Add payment method
- [ ] Prepay 3 months: $78
- [ ] **Create Project:**
  - [ ] Name: REPZ
  - [ ] Platform: React
- [ ] **Get DSN:**
  - [ ] Copy DSN: https://________________@sentry.io/________________
- [ ] **Cost:** $78 ✅

### 3.3 Cal.com Pro - $16 (1 month)
- [ ] Open: https://cal.com/signup
- [ ] Sign up
- [ ] Select Pro plan ($16/month)
- [ ] Add payment method: $16
- [ ] **Set up booking page:**
  - [ ] Event type: "Discovery Call"
  - [ ] Duration: 30 minutes
  - [ ] Connect Google Calendar
- [ ] **Get booking link:** cal.com/________________
- [ ] **Cost:** $16 ✅

**Phase 3 Total: $130 | Time: _____ minutes | Running Total: $340**

**Reserve: $40 for API overages and surprises**

---

## ✅ Phase 4: Environment Configuration (5 minutes)

### 4.1 Create .env.production file
- [ ] Copy template from `.env.production.template`
- [ ] Fill in all values from above phases
- [ ] **Verify all required variables:**
  - [ ] VITE_SUPABASE_URL
  - [ ] VITE_SUPABASE_ANON_KEY
  - [ ] SUPABASE_SERVICE_ROLE_KEY
  - [ ] VITE_STRIPE_PUBLIC_KEY
  - [ ] STRIPE_SECRET_KEY
  - [ ] STRIPE_WEBHOOK_SECRET
  - [ ] SENDGRID_API_KEY
  - [ ] VITE_SENTRY_DSN
  - [ ] VITE_APP_URL (your Vercel URL)

### 4.2 Configure Vercel Environment Variables
- [ ] Go to Vercel Dashboard → Settings → Environment Variables
- [ ] Add all variables from .env.production
- [ ] Select: Production, Preview, Development (all environments)
- [ ] Save all variables

**Phase 4 Total: 5 minutes | Time: _____ minutes**

---

## ✅ Phase 5: Database Setup (5 minutes)

### 5.1 Apply Referral System Migration
- [ ] Go to Supabase Dashboard → SQL Editor
- [ ] Open file: `supabase/migrations/20251118000000_referral_system.sql`
- [ ] Copy entire contents
- [ ] Paste into SQL Editor
- [ ] Run query
- [ ] **Verify tables created:**
  - [ ] `referral_codes`
  - [ ] `referrals`
  - [ ] `referral_rewards`

### 5.2 Verify Existing Tables
- [ ] Check Table Editor
- [ ] Confirm tables exist:
  - [ ] `non_portal_clients`
  - [ ] `workouts`
  - [ ] `nutrition_plans`
  - [ ] (other required tables)

**Phase 5 Total: 5 minutes | Time: _____ minutes**

---

## ✅ Phase 6: DEPLOY! (10 minutes)

### 6.1 Build and Deploy
```bash
cd /home/user/alaweimm90-business/repz/REPZ/platform

# Verify build works locally
npm run build:production

# Deploy to Vercel
npx vercel --prod

# Or use our script
npm run deploy:production
```

- [ ] Build completes successfully
- [ ] Deployment successful
- [ ] **Note deployment URL:** __________________

### 6.2 Connect Custom Domain
- [ ] Go to Vercel → Settings → Domains
- [ ] Add domain from 1.3
- [ ] Add DNS records to Namecheap:
  - [ ] A record: 76.76.21.21
  - [ ] CNAME: cname.vercel-dns.com
- [ ] Wait for SSL certificate (1-2 minutes)
- [ ] **Verify:** https://yourdomain.com loads

**Phase 6 Total: 10 minutes | Time: _____ minutes**

---

## ✅ Phase 7: TEST EVERYTHING (10 minutes)

### 7.1 Intake Form Test
- [ ] Go to: https://yourdomain.com/intake
- [ ] Fill out all 7 steps:
  - [ ] Step 1: Account details
  - [ ] Step 2: Profile
  - [ ] Step 3: Health info
  - [ ] Step 4: Training preferences
  - [ ] Step 5: Nutrition preferences
  - [ ] Step 6: Goals
  - [ ] Step 7: Payment selection
- [ ] Select "REPZ Basic" ($299)
- [ ] **Use Stripe test card:** 4242 4242 4242 4242
- [ ] Complete payment
- [ ] **Verify:**
  - [ ] Payment successful
  - [ ] Confirmation email received (check spam)
  - [ ] Admin email received
  - [ ] No errors in Sentry

### 7.2 Admin Dashboard Test
- [ ] Log in as admin
- [ ] Go to: https://yourdomain.com/admin/non-portal-clients
- [ ] **Verify:**
  - [ ] Test client appears
  - [ ] Payment status: "paid"
  - [ ] All intake data visible
  - [ ] Can add notes

### 7.3 Error Monitoring Test
- [ ] Go to Sentry dashboard
- [ ] **Verify:**
  - [ ] No critical errors
  - [ ] Only test transaction visible
  - [ ] Performance metrics showing

**Phase 7 Total: 10 minutes | Time: _____ minutes**

---

## ✅ Phase 8: GO LIVE! (5 minutes)

### 8.1 Switch to Live Mode
- [ ] Stripe: Go to Developers → toggle to "Live mode"
- [ ] Update Vercel env vars with live Stripe keys
- [ ] Redeploy: `npx vercel --prod`

### 8.2 Create Admin Account
- [ ] Sign up at: https://yourdomain.com/signup
- [ ] Use: admin@yourdomain.com
- [ ] Verify email
- [ ] Set as admin in Supabase (if needed)

### 8.3 Share & Launch! 🎉
- [ ] Post on social media
- [ ] Share booking link: cal.com/yourname
- [ ] Send to beta users
- [ ] Update website/landing page

**Phase 8 Total: 5 minutes | Time: _____ minutes**

---

## 📊 Final Checklist

**Pre-Launch:**
- [ ] All tests passing
- [ ] No console errors
- [ ] Environment variables set
- [ ] Stripe webhook working
- [ ] Email sending
- [ ] Database accessible
- [ ] Domain connected
- [ ] SSL certificate active
- [ ] Analytics tracking
- [ ] Error monitoring active

**Post-Launch:**
- [ ] Monitor Sentry for errors (first 24 hours)
- [ ] Check email deliverability
- [ ] Verify payment flows
- [ ] Monitor Stripe dashboard
- [ ] Check Vercel analytics
- [ ] Test on mobile devices
- [ ] Share with first beta users

---

## 💰 Budget Tracking

| Service | Cost | Status |
|---------|------|--------|
| Vercel Pro (3mo) | $60 | ☐ |
| Supabase Pro (3mo) | $75 | ☐ |
| Domain | $15 | ☐ |
| Stripe | $0 | ☐ |
| SendGrid (3mo) | $60 | ☐ |
| Google Workspace (3mo) | $36 | ☐ |
| Sentry (3mo) | $78 | ☐ |
| Cal.com (1mo) | $16 | ☐ |
| **Total Spent** | **$340** | |
| **Reserve** | **$40** | |
| **Grand Total** | **$380** | |

---

## 🆘 Troubleshooting

**Build fails:**
- Check all environment variables are set
- Run `npm install` again
- Clear node_modules: `rm -rf node_modules && npm install`

**Deployment fails:**
- Verify Vercel project connected to correct repo
- Check build logs in Vercel dashboard
- Ensure all env vars added

**Payments not working:**
- Verify Stripe webhook URL is correct
- Check webhook signing secret
- Test with Stripe test card first

**Emails not sending:**
- Verify SendGrid domain authenticated
- Check API key has full permissions
- Look for emails in spam folder

**Domain not connecting:**
- DNS can take up to 48 hours (usually 5-10 minutes)
- Verify A and CNAME records correct
- Check SSL certificate status in Vercel

---

## 📞 Support

- **Vercel:** https://vercel.com/support
- **Supabase:** https://supabase.com/support
- **Stripe:** https://support.stripe.com
- **SendGrid:** https://support.sendgrid.com
- **Sentry:** https://sentry.io/support

---

**Deployment completed at:** __________

**First paying client:** __________

**Launch post:** __________

🎉 **REPZ IS LIVE!** 🎉
