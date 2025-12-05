# Quick Start - REPZCoach Launch

## Immediate Actions (Today)

### 1. Stripe Setup (15 min)

1. Go to https://dashboard.stripe.com/products
2. Create 3 products:
   - "REPZ Basic Plan" - $299 one-time payment
   - "REPZ Premium Plan" - $599 one-time payment
   - "REPZ Concierge" - $1,499 one-time payment
3. Copy the 3 price IDs (start with `price_`)

### 2. Supabase Edge Functions (Manual Deploy)

Since Supabase CLI isn't installed, deploy via Supabase Dashboard:

1. Go to https://supabase.com/dashboard/project/[your-project]/functions
2. Create 3 new functions:

**Function 1: `create-intake-checkout`**

- Copy code from: `Repz/supabase/functions/create-intake-checkout/index.ts`
- Click "Deploy"

**Function 2: `deliver-plan`**

- Copy code from: `Repz/supabase/functions/deliver-plan/index.ts`
- Click "Deploy"

**Function 3: Update `stripe-webhook`**

- Copy updated code from: `Repz/supabase/functions/stripe-webhook/index.ts`
- Click "Deploy"

### 3. Environment Variables

In Supabase Dashboard → Settings → Edge Functions → Secrets:

```bash
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_BASIC=price_...
STRIPE_PRICE_PREMIUM=price_...
STRIPE_PRICE_CONCIERGE=price_...
SENDGRID_API_KEY=SG....
```

### 4. Database Migration

In Supabase Dashboard → SQL Editor:

- Run the SQL from: `Repz/supabase/migrations/20250112000001_create_non_portal_clients.sql`

### 5. Stripe Webhook

1. Go to https://dashboard.stripe.com/webhooks
2. Add endpoint: `https://[project-ref].supabase.co/functions/v1/stripe-webhook`
3. Select events:
   - `checkout.session.completed`
   - `payment_intent.succeeded`
   - `payment_intent.payment_failed`
4. Copy webhook signing secret

### 6. Add Routes to App

Update your router file to include:

```typescript
import IntakeLanding from '@/pages/IntakeLanding';
import IntakeEmail from '@/pages/IntakeEmail';
import NonPortalClients from '@/pages/admin/NonPortalClients';

// Add these routes
<Route path="/intake" element={<IntakeLanding />} />
<Route path="/intake-email" element={<IntakeEmail />} />
<Route path="/admin/non-portal-clients" element={<NonPortalClients />} />
```

### 7. Test Flow

1. Visit `/intake` landing page
2. Click "Start Your Intake Form"
3. Fill out all 7 steps
4. Select a plan
5. Use Stripe test card: `4242 4242 4242 4242`
6. Verify email received
7. Check admin dashboard at `/admin/non-portal-clients`

## Files Created/Modified

### New Files:

- `Repz/supabase/functions/create-intake-checkout/index.ts`
- `Repz/supabase/functions/deliver-plan/index.ts`
- `Repz/REPZ/platform/src/pages/IntakeLanding.tsx`
- `Repz/REPZ_SETUP_GUIDE.md`

### Modified Files:

- `Repz/REPZ/platform/src/pages/IntakeEmail.tsx` (added Stripe checkout)
- `Repz/REPZ/platform/src/pages/admin/NonPortalClients.tsx` (added delivery modal)
- `Repz/supabase/functions/stripe-webhook/index.ts` (added intake handling)

## Next Steps After Launch

### Week 1:

- [ ] Get first test client through full flow
- [ ] Create plan template (Google Doc or PDF)
- [ ] Upload to storage (Supabase Storage or Google Drive)
- [ ] Test plan delivery

### Week 2:

- [ ] Market on social media
- [ ] Reach out to 10 potential clients
- [ ] Refine intake form based on feedback
- [ ] Build plan template library

### Week 3:

- [ ] Process 5 real clients
- [ ] Calculate time per plan
- [ ] Consider automation/AI for plan generation
- [ ] Add upsell to monthly coaching

## Troubleshooting

**Edge functions not deploying?**

- Check function logs in Supabase Dashboard
- Verify all environment variables are set
- Test locally with `supabase functions serve`

**Stripe checkout not working?**

- Verify price IDs match environment variables
- Check Stripe logs for errors
- Ensure webhook is receiving events

**Emails not sending?**

- Verify SendGrid API key
- Check sender email is verified in SendGrid
- Review Edge function logs

## Support Checklist

Before asking for help:

- [ ] Checked Supabase function logs
- [ ] Checked Stripe webhook logs
- [ ] Verified all environment variables
- [ ] Tested with Stripe test mode
- [ ] Reviewed browser console for errors

---

**Ready to launch?** Start with Stripe setup, then deploy Edge functions, then test!
