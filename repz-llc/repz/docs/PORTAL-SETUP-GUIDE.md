# REPZ Portal Setup Guide

Complete guide to set up the REPZ platform for production use, including creating admin and client accounts.

---

## Prerequisites

1. **Supabase Project** - Active project at [supabase.com](https://supabase.com)
2. **Stripe Account** - With products/prices configured
3. **Node.js 18+** - For local development
4. **Environment Variables** - Configured in `.env.local`

---

## Step 1: Environment Configuration

### 1.1 Copy Environment File

```bash
cp .env.example .env.local
```

### 1.2 Configure Supabase

Get these from your Supabase project dashboard:

```env
VITE_SUPABASE_URL=https://lvmcumsfpjjcgnnovvzs.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 1.3 Configure Stripe

```env
STRIPE_SECRET_KEY=sk_live_... (or sk_test_... for testing)
STRIPE_WEBHOOK_SECRET=whsec_...
```

### 1.4 Verify Price IDs

Ensure all tier price IDs are set (see `.env.example` for full list):

```env
STRIPE_PRICE_CORE_MONTHLY_PROD=price_1QkCJpAb5lqTiGnqjjW6WSnF
STRIPE_PRICE_ADAPTIVE_MONTHLY_PROD=price_1QkCKZAb5lqTiGnqTSQgRrBh
STRIPE_PRICE_PERFORMANCE_MONTHLY_PROD=price_1QkCLBAb5lqTiGnqLgZWL8E0
STRIPE_PRICE_LONGEVITY_MONTHLY_PROD=price_1QkCLnAb5lqTiGnqzfAKPyim
```

---

## Step 2: Database Setup

### 2.1 Run Migrations

```bash
npx supabase db push
```

Or apply migrations manually in Supabase SQL Editor.

### 2.2 Verify Tables Exist

Check these tables exist in your Supabase database:

- `profiles` - User profiles with subscription info
- `user_roles` - Role assignments (admin, coach, client)
- `client_profiles` - Detailed client information
- `coach_profiles` - Coach information
- `non_portal_clients` - Email-based intake submissions

---

## Step 3: Create Admin Account

### Option A: Via Supabase Dashboard (Recommended)

1. Go to **Supabase Dashboard** → **Authentication** → **Users**
2. Click **Add User**
3. Enter:
   - Email: `admin@repzcoach.com` (or your email)
   - Password: Strong password (8+ chars, uppercase, lowercase, number)
4. Click **Create User**
5. Copy the **User ID** (UUID)

### Option B: Via SignUp Page

1. Go to `http://localhost:8080/signup`
2. Create account with your admin email
3. Verify email if required

### 3.1 Assign Admin Role

Run in Supabase SQL Editor:

```sql
-- Replace with actual user ID from step above
DO $$
DECLARE
  admin_id UUID := 'YOUR-ADMIN-USER-ID-HERE';
BEGIN
  -- Update profile to highest tier
  UPDATE public.profiles 
  SET subscription_tier = 'longevity',
      subscription_status = 'active'
  WHERE id = admin_id;

  -- Assign admin role
  INSERT INTO public.user_roles (user_id, role)
  VALUES (admin_id, 'admin')
  ON CONFLICT (user_id, role) DO NOTHING;
  
  RAISE NOTICE 'Admin role assigned to user %', admin_id;
END $$;
```

---

## Step 4: Create Emilio's Account (Performance Tier)

### 4.1 Create User

1. Go to **Supabase Dashboard** → **Authentication** → **Users**
2. Click **Add User**
3. Enter:
   - Email: Emilio's email address
   - Password: Temporary password (he can reset later)
4. Click **Create User**
5. Copy the **User ID** (UUID)

### 4.2 Configure Emilio's Profile

Run in Supabase SQL Editor:

```sql
-- Replace with actual user ID and email
DO $$
DECLARE
  emilio_id UUID := 'EMILIOS-USER-ID-HERE';
  emilio_email TEXT := 'emilio@example.com';
BEGIN
  -- Update profile to Performance tier (3rd tier)
  UPDATE public.profiles 
  SET subscription_tier = 'performance',
      subscription_status = 'active',
      email = emilio_email
  WHERE id = emilio_id;

  -- Ensure client role
  INSERT INTO public.user_roles (user_id, role)
  VALUES (emilio_id, 'client')
  ON CONFLICT (user_id, role) DO NOTHING;

  -- Create client profile for coaching features
  INSERT INTO public.client_profiles (
    auth_user_id,
    client_name,
    subscription_tier,
    onboarding_completed
  )
  VALUES (
    emilio_id,
    'Emilio',
    'performance',
    false
  )
  ON CONFLICT (auth_user_id) DO UPDATE SET
    subscription_tier = 'performance';
  
  RAISE NOTICE 'Emilio configured with Performance tier';
END $$;
```

---

## Step 5: Deploy Edge Functions

### 5.1 Deploy All Functions

```bash
npx supabase functions deploy
```

### 5.2 Set Function Secrets

```bash
# Stripe
npx supabase secrets set STRIPE_SECRET_KEY=sk_live_...
npx supabase secrets set STRIPE_WEBHOOK_SECRET=whsec_...

# Price IDs
npx supabase secrets set STRIPE_PRICE_CORE_MONTHLY_PROD=price_...
npx supabase secrets set STRIPE_PRICE_ADAPTIVE_MONTHLY_PROD=price_...
npx supabase secrets set STRIPE_PRICE_PERFORMANCE_MONTHLY_PROD=price_...
npx supabase secrets set STRIPE_PRICE_LONGEVITY_MONTHLY_PROD=price_...

# OpenAI (for AI features)
npx supabase secrets set OPENAI_API_KEY=sk-...
```

### 5.3 Verify Functions

Check functions are deployed:

```bash
npx supabase functions list
```

Key functions to verify:
- `create-checkout` - Stripe checkout sessions
- `stripe-webhook` - Payment webhook handling
- `auth-register` - User registration
- `check-subscription` - Subscription verification

---

## Step 6: Configure Stripe Webhook

### 6.1 Create Webhook Endpoint

1. Go to **Stripe Dashboard** → **Developers** → **Webhooks**
2. Click **Add Endpoint**
3. Enter URL: `https://lvmcumsfpjjcgnnovvzs.supabase.co/functions/v1/stripe-webhook`
4. Select events:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`
5. Click **Add Endpoint**
6. Copy the **Signing Secret** and add to Supabase secrets

---

## Step 7: Start Development Server

```bash
npm run dev
```

Access at: `http://localhost:8080`

---

## Step 8: Verify Setup

### 8.1 Test Admin Login

1. Go to `http://localhost:8080/login`
2. Login with admin credentials
3. Verify you see the dashboard
4. Check admin features are accessible

### 8.2 Test Emilio's Login

1. Go to `http://localhost:8080/login`
2. Login with Emilio's credentials
3. Verify Performance tier features are visible
4. Check intake form is accessible

### 8.3 Test Intake Flow

1. Go to `http://localhost:8080/intake`
2. Complete the intake form
3. Verify data saves correctly

### 8.4 Test Payment Flow (Test Mode)

1. Login as a new user
2. Go to pricing page
3. Select a tier
4. Complete checkout with Stripe test card: `4242 4242 4242 4242`
5. Verify subscription is created

---

## Tier Access Matrix

| Feature | Core | Adaptive | Performance | Longevity |
|---------|------|----------|-------------|-----------|
| Training Program | ✅ | ✅ | ✅ | ✅ |
| Nutrition Plan | ✅ | ✅ | ✅ | ✅ |
| Response Time | 72h | 48h | 24h | 12h |
| Biomarkers | ❌ | ✅ | ✅ | ✅ |
| Wearable Integration | ❌ | ✅ | ✅ | ✅ |
| Weekly Check-ins | ❌ | ✅ | ✅ | ✅ |
| AI Assistant | ❌ | ❌ | ✅ | ✅ |
| Form Analysis | ❌ | ❌ | ✅ | ✅ |
| PEDs Protocols | ❌ | ❌ | ✅ | ✅ |
| In-Person Training | ❌ | ❌ | ❌ | ✅ |
| Concierge Service | ❌ | ❌ | ❌ | ✅ |

---

## Troubleshooting

### "Invalid login credentials"

- Verify email is confirmed in Supabase Auth
- Check password meets requirements
- Ensure user exists in `auth.users`

### "Subscription not found"

- Check `profiles` table has correct `subscription_tier`
- Verify `subscription_status` is 'active'
- Run profile update SQL if needed

### Edge function errors

- Check function logs: `npx supabase functions logs <function-name>`
- Verify all secrets are set
- Check CORS headers in function

### Stripe webhook not working

- Verify webhook URL is correct
- Check signing secret matches
- Test with Stripe CLI: `stripe listen --forward-to localhost:54321/functions/v1/stripe-webhook`

---

## Quick Reference

### URLs

| Environment | URL |
|-------------|-----|
| Local Dev | http://localhost:8080 |
| Supabase Studio | https://supabase.com/dashboard/project/lvmcumsfpjjcgnnovvzs |
| Stripe Dashboard | https://dashboard.stripe.com |

### Key Routes

| Route | Purpose |
|-------|---------|
| `/login` | User login |
| `/signup` | User registration |
| `/dashboard` | Main dashboard |
| `/intake` | Intake form landing |
| `/intake-email` | Email-based intake |
| `/pricing` | Subscription pricing |
| `/coach-admin` | Coach admin panel |

### Tier IDs

Always use these exact tier IDs:
- `core` - $89/month
- `adaptive` - $149/month
- `performance` - $229/month (Emilio's tier)
- `longevity` - $349/month

---

## Support

- **Documentation:** See `/docs` folder
- **Issues:** Check GitHub issues
- **Logs:** Supabase Dashboard → Logs

---

*Last Updated: December 2025*
