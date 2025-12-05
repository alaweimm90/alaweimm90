# REPZCoach Deployment Checklist

## Pre-Deployment

### Code Changes

- [x] Created 3 Edge functions
- [x] Updated stripe-webhook handler
- [x] Created IntakeLanding page
- [x] Updated IntakeEmail with Stripe checkout
- [x] Enhanced NonPortalClients admin page
- [x] Added routes to App.tsx

### Database

- [ ] Run migration: `20250112000001_create_non_portal_clients.sql`

### Environment Variables (Supabase)

- [ ] STRIPE_SECRET_KEY
- [ ] STRIPE_WEBHOOK_SECRET
- [ ] STRIPE_PRICE_BASIC
- [ ] STRIPE_PRICE_PREMIUM
- [ ] STRIPE_PRICE_CONCIERGE
- [ ] SENDGRID_API_KEY

## Stripe Setup

### Products

- [ ] Create "REPZ Basic Plan" - $299
- [ ] Create "REPZ Premium Plan" - $599
- [ ] Create "REPZ Concierge" - $1,499
- [ ] Copy 3 price IDs

### Webhook

- [ ] Add endpoint URL
- [ ] Select events (checkout.session.completed, payment_intent.\*)
- [ ] Copy webhook secret

## Supabase Deployment

### Edge Functions

- [ ] Deploy `create-intake-checkout`
- [ ] Deploy `deliver-plan`
- [ ] Update `stripe-webhook`
- [ ] Test function logs

### Database

- [ ] Run migration SQL
- [ ] Verify table created
- [ ] Check RLS policies

## Testing

### Local Test

- [ ] Visit `/intake`
- [ ] Complete intake form
- [ ] Use test card: 4242 4242 4242 4242
- [ ] Verify redirect to success
- [ ] Check email received

### Admin Test

- [ ] Login as admin
- [ ] Visit `/admin/non-portal-clients`
- [ ] View submitted intake
- [ ] Test plan delivery modal

### Production Test

- [ ] Repeat local test on production
- [ ] Verify webhook receives events
- [ ] Check Stripe dashboard
- [ ] Verify emails sent

## Post-Deployment

### Monitoring

- [ ] Check Supabase function logs
- [ ] Monitor Stripe webhook logs
- [ ] Track email delivery rates

### First Client

- [ ] Share `/intake` link
- [ ] Process first real submission
- [ ] Create and deliver plan
- [ ] Collect feedback

## Rollback Plan

If issues occur:

1. Disable webhook in Stripe
2. Revert Edge function deployments
3. Remove routes from App.tsx
4. Investigate logs

## Support Resources

- Supabase Dashboard: https://supabase.com/dashboard
- Stripe Dashboard: https://dashboard.stripe.com
- SendGrid Dashboard: https://app.sendgrid.com
- Function Logs: Supabase → Functions → Logs
- Webhook Logs: Stripe → Webhooks → Events
