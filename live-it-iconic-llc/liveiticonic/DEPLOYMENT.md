# 🚀 LiveItIconic.com - Production Deployment Guide

**Technical Partner: Claude (AI)**
**CEO/Founder: [Partner Name]**
**Partnership Structure: 50/50 Technical/Business Split**

---

## 📋 Pre-Deployment Checklist

### 1. Domain Setup
- [ ] Purchase domain: `liveiticonic.com`
- [ ] Configure DNS with hosting provider
- [ ] Set up SSL certificate (Let's Encrypt recommended)
- [ ] Configure www redirect (www.liveiticonic.com → liveiticonic.com)

### 2. Supabase Setup
- [ ] Create Supabase project at https://supabase.com
- [ ] Run all migrations in `supabase/migrations/` folder
- [ ] Enable Row Level Security (RLS) on all tables
- [ ] Create storage buckets:
  - `product-images`
  - `review-images`
  - `brand-assets`
- [ ] Configure email templates
- [ ] Set up authentication providers (Email, Google OAuth optional)

### 3. Stripe Setup
- [ ] Create Stripe account at https://stripe.com
- [ ] Complete business verification
- [ ] Set up products in Stripe Dashboard:
  - Black Hoodie - $79.00
  - Black Cap - $29.00
  - Black T-Shirt - $49.00
- [ ] Configure shipping rates
- [ ] Set up webhook endpoint: `https://liveiticonic.com/api/webhooks/stripe`
- [ ] Enable payment methods: Card, Apple Pay, Google Pay

### 4. Email Service (Resend)
- [ ] Create account at https://resend.com
- [ ] Verify domain for sending emails
- [ ] Configure email templates for:
  - Order confirmations
  - Shipping notifications
  - Abandoned cart recovery
  - Newsletter

### 5. Analytics & Monitoring
- [ ] Set up Google Analytics 4
- [ ] Configure Google Tag Manager
- [ ] Set up Facebook Pixel (optional)
- [ ] Install Sentry for error tracking
- [ ] Configure uptime monitoring (UptimeRobot recommended)

---

## 🔐 Environment Variables

Create `.env.production` file with the following:

```bash
# ===================================
# Supabase Configuration
# ===================================
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key-here
VITE_SUPABASE_PUBLISHABLE_KEY=your-publishable-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here

# ===================================
# Stripe Configuration
# ===================================
VITE_STRIPE_PUBLIC_KEY=pk_live_your-publishable-key-here
STRIPE_SECRET_KEY=sk_live_your-secret-key-here
STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret-here

# ===================================
# Email Configuration (Resend)
# ===================================
RESEND_API_KEY=re_your-api-key-here
EMAIL_FROM=orders@liveiticonic.com
EMAIL_SUPPORT=support@liveiticonic.com

# ===================================
# Application Configuration
# ===================================
VITE_APP_URL=https://liveiticonic.com
VITE_APP_NAME="Live It Iconic"
NODE_ENV=production

# ===================================
# JWT Configuration
# ===================================
JWT_SECRET=your-super-secret-jwt-key-minimum-32-characters-long
JWT_EXPIRATION=7d

# ===================================
# Analytics
# ===================================
VITE_GA_MEASUREMENT_ID=G-XXXXXXXXXX
VITE_GTM_ID=GTM-XXXXXXX
VITE_FACEBOOK_PIXEL_ID=your-pixel-id

# ===================================
# Sentry (Error Tracking)
# ===================================
VITE_SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# ===================================
# Social Media API Keys (Optional)
# ===================================
YOUTUBE_API_KEY=your-youtube-api-key
TWITCH_CLIENT_ID=your-twitch-client-id
TWITCH_CLIENT_SECRET=your-twitch-client-secret
```

### 🔒 Security Notes:
- **NEVER** commit `.env.production` to git
- Use environment variable management in your hosting provider
- Rotate keys every 90 days
- Use different keys for staging and production

---

## 🏗️ Build & Deployment

### Option 1: Vercel (Recommended for speed)

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy
vercel --prod

# Configure environment variables in Vercel Dashboard
# Settings → Environment Variables → Add all .env.production variables
```

**Vercel Configuration** (`vercel.json`):
```json
{
  "framework": "vite",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "installCommand": "npm install --legacy-peer-deps",
  "devCommand": "npm run dev",
  "regions": ["iad1"],
  "env": {
    "NODE_ENV": "production"
  }
}
```

### Option 2: Netlify

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Deploy
netlify deploy --prod

# Configure environment variables in Netlify Dashboard
```

**Netlify Configuration** (`netlify.toml`):
```toml
[build]
  command = "npm run build"
  publish = "dist"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200

[build.environment]
  NODE_ENV = "production"
```

### Option 3: Custom Server (VPS/AWS/DigitalOcean)

```bash
# Build for production
npm run build

# Serve with a static file server
npm install -g serve
serve -s dist -l 3000

# Or use Nginx/Apache
```

---

## 📊 Database Migrations

### Apply Migrations to Supabase

1. **Via Supabase Dashboard**:
   - Go to SQL Editor in Supabase Dashboard
   - Copy contents of each migration file
   - Run in order: 001 → 002 → 003 → 004

2. **Via Supabase CLI** (Recommended):
```bash
# Install Supabase CLI
npm install -g supabase

# Login
supabase login

# Link project
supabase link --project-ref your-project-ref

# Push migrations
supabase db push
```

### Migration Order:
1. `001_initial_schema.sql` - User profiles and addresses
2. `002_products_inventory.sql` - Products and inventory system
3. `003_orders_payments.sql` - Orders, payments, discount codes
4. `004_reviews_ratings.sql` - Reviews and rating system

---

## 🛍️ Initial Data Setup

### 1. Create Product Categories

```sql
INSERT INTO public.categories (name, slug, description) VALUES
('Hoodies', 'hoodies', 'Premium lifestyle hoodies'),
('Caps', 'caps', 'Stylish caps for automotive enthusiasts'),
('T-Shirts', 't-shirts', 'Comfortable lifestyle t-shirts');
```

### 2. Create Products

```sql
-- Black Hoodie
INSERT INTO public.products (
  name,
  slug,
  description,
  price,
  category_id,
  sku,
  is_active,
  is_featured
) VALUES (
  'Live It Iconic Black Hoodie',
  'black-hoodie',
  'Premium quality black hoodie for automotive enthusiasts. Features iconic design and superior comfort.',
  7900, -- $79.00 in cents
  (SELECT id FROM public.categories WHERE slug = 'hoodies'),
  'LII-HOODIE-BLK',
  TRUE,
  TRUE
);

-- Black Cap
INSERT INTO public.products (
  name,
  slug,
  description,
  price,
  category_id,
  sku,
  is_active,
  is_featured
) VALUES (
  'Live It Iconic Black Cap',
  'black-cap',
  'Stylish black cap with embroidered logo. Perfect for everyday wear.',
  2900, -- $29.00 in cents
  (SELECT id FROM public.categories WHERE slug = 'caps'),
  'LII-CAP-BLK',
  TRUE,
  TRUE
);

-- Black T-Shirt
INSERT INTO public.products (
  name,
  slug,
  description,
  price,
  category_id,
  sku,
  is_active,
  is_featured
) VALUES (
  'Live It Iconic Black T-Shirt',
  'black-tshirt',
  'Classic black t-shirt with iconic branding. Made from premium cotton blend.',
  4900, -- $49.00 in cents
  (SELECT id FROM public.categories WHERE slug = 't-shirts'),
  'LII-TEE-BLK',
  TRUE,
  TRUE
);
```

### 3. Set Initial Inventory

```sql
-- Add inventory for each product
INSERT INTO public.inventory (product_id, quantity, low_stock_threshold) VALUES
((SELECT id FROM public.products WHERE sku = 'LII-HOODIE-BLK'), 100, 10),
((SELECT id FROM public.products WHERE sku = 'LII-CAP-BLK'), 150, 15),
((SELECT id FROM public.products WHERE sku = 'LII-TEE-BLK'), 200, 20);
```

---

## 🎯 Post-Deployment Tasks

### Immediate (Day 1)
- [ ] Test complete purchase flow (staging card: 4242 4242 4242 4242)
- [ ] Verify email confirmations are sent
- [ ] Test user registration and login
- [ ] Verify product images load correctly
- [ ] Test mobile responsiveness
- [ ] Set up Google Search Console
- [ ] Submit sitemap to Google
- [ ] Test Stripe webhook is working

### Week 1
- [ ] Monitor error logs in Sentry
- [ ] Check analytics data is flowing
- [ ] Test abandoned cart email triggers
- [ ] Verify inventory tracking works
- [ ] Set up backup schedule for Supabase
- [ ] Create admin user account
- [ ] Test discount code system

### Week 2
- [ ] Performance audit with Lighthouse
- [ ] SEO optimization check
- [ ] Set up uptime monitoring alerts
- [ ] Create customer support workflow
- [ ] Test review submission and moderation
- [ ] Verify shipping calculations

---

## 🎨 Social Media Setup

### YouTube Channel Setup
1. Create YouTube channel: "Live It Iconic"
2. Upload brand assets (logo, banner)
3. Create channel trailer
4. Upload initial content:
   - Brand story
   - Product showcases
   - Behind-the-scenes
   - Automotive lifestyle content

### Twitch Channel Setup
1. Create Twitch account: "LiveItIconic"
2. Set up stream layout and overlays
3. Create stream schedule
4. Plan content:
   - Product launches
   - Community events
   - Automotive discussions
   - Live Q&A sessions

### Instagram Setup
1. Create business account: @liveiticonic
2. Link to Facebook Page
3. Set up Instagram Shopping
4. Create content calendar
5. Plan posts:
   - Product photos
   - Lifestyle shots
   - User-generated content
   - Behind-the-scenes

### TikTok Setup
1. Create business account: @liveiticonic
2. Set up TikTok Shop
3. Create short-form content
4. Engage with automotive community

---

## 📈 Marketing Launch Strategy

### Phase 1: Soft Launch (Week 1-2)
- [ ] Announce on personal social media
- [ ] Reach out to automotive influencers
- [ ] Create launch discount code: LAUNCH20 (20% off)
- [ ] Send to friends and family for testing
- [ ] Collect initial reviews

### Phase 2: Official Launch (Week 3-4)
- [ ] Press release to automotive blogs
- [ ] Paid advertising on Instagram/Facebook
- [ ] YouTube video announcement
- [ ] Email newsletter to waitlist
- [ ] Reddit posts in automotive communities
- [ ] Collaboration with car meet organizers

### Phase 3: Growth (Month 2+)
- [ ] Influencer partnerships
- [ ] User-generated content campaigns
- [ ] Seasonal promotions
- [ ] Product line expansion
- [ ] Community building events
- [ ] Referral program

---

## 🛠️ Maintenance & Monitoring

### Daily
- Check error logs in Sentry
- Monitor sales dashboard
- Respond to customer inquiries
- Check inventory levels

### Weekly
- Review analytics trends
- Update social media content
- Process reviews and approvals
- Backup database
- Check website performance

### Monthly
- Review and analyze sales data
- Plan marketing campaigns
- Update product inventory
- Review customer feedback
- Update product photos/descriptions

---

## 🆘 Troubleshooting

### Common Issues

**Stripe webhook not working:**
- Verify webhook URL in Stripe Dashboard
- Check webhook secret matches `.env.production`
- Test with Stripe CLI: `stripe listen --forward-to localhost:3000/api/webhooks/stripe`

**Emails not sending:**
- Verify Resend API key
- Check domain verification in Resend
- Ensure FROM email is verified
- Check spam folder

**Products not showing:**
- Verify products have `is_active = TRUE`
- Check RLS policies are enabled
- Ensure images are uploaded to Supabase Storage

**Checkout failing:**
- Test with Stripe test card: 4242 4242 4242 4242
- Check Stripe API keys (pk_live vs pk_test)
- Verify payment intent creation in Stripe logs

---

## 📞 Support Contacts

**Technical Issues:**
- Sentry: Real-time error tracking
- GitHub: Issue tracking
- Supabase Support: Database issues

**Business/Marketing:**
- Email: info@liveiticonic.com
- Phone: [Your phone]
- Social: @liveiticonic

---

## 🎉 Success Metrics

### Launch Goals (First 30 Days)
- 100+ email subscribers
- 50+ social media followers per platform
- 10+ product sales
- $1,000+ revenue
- 5+ customer reviews

### 6-Month Goals
- 1,000+ email subscribers
- 1,000+ Instagram followers
- 100+ product sales
- $10,000+ revenue
- 50+ customer reviews
- 4.5+ star average rating

---

**Last Updated:** November 2025
**Version:** 1.0.0
**Status:** Production Ready ✅

For technical partnership questions, refer to PARTNERSHIP.md
