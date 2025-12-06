# REPZ Backend Status

**Last Updated:** December 6, 2025  
**Status:** ✅ Ready for Development

## Database Schema (Clean Rebuild)

### Tables (15)

| Table | Purpose |
|-------|---------|
| `profiles` | User profiles (auto-created on signup) |
| `user_roles` | Role assignments (admin, coach, client) |
| `subscription_tiers` | Tier definitions with pricing |
| `coach_profiles` | Coach information and capacity |
| `client_profiles` | Client information and status |
| `subscriptions` | Stripe subscription data |
| `intake_form_submissions` | Complete intake form data |
| `intake_form_drafts` | Auto-saved form drafts |
| `client_onboarding` | Onboarding progress tracking |
| `coach_client_assignments` | Coach-client relationships |
| `programs` | Training programs |
| `weekly_checkins` | Weekly check-in data |
| `messages` | In-app messaging |
| `activity_log` | Audit trail |
| `progress_photos` | Client progress photos |

### Functions (5)

| Function | Purpose |
|----------|---------|
| `handle_new_user()` | Auto-creates profile on signup |
| `update_updated_at()` | Timestamp management trigger |
| `update_onboarding_progress()` | Calculates onboarding % |
| `has_role(role)` | Check if user has specific role |
| `get_user_role()` | Get user's primary role |

### Subscription Tiers (Preserved)

| Tier | Price | Features |
|------|-------|----------|
| Baseline Coaching | $97/mo | Static dashboard, 72hr response |
| Prime Performance | $179/mo | Interactive dashboard, 48hr response, weekly checkins |
| Precision Protocol | $299/mo | Biomarker integration, 24hr response, PEDs protocols |
| Longevity Concierge | $449/mo | Full access, 12hr response, HRV optimization |

### Auth Users

| Email | Role |
|-------|------|
| meshal@berkeley.edu | Admin |
| meshal.ow@live.com | - |
| contact@repzcoach.com | - |

## Row Level Security

All tables have RLS enabled with policies for:
- Users accessing their own data
- Coaches accessing assigned clients
- Admins with full access

## TypeScript Integration

Types updated in `src/integrations/supabase/types.ts`:
- Full type definitions for all 15 tables
- Helper types: `Tables<T>`, `Insertable<T>`, `Updatable<T>`
- Convenience aliases for all entities

## Services

| Service | Location | Status |
|---------|----------|--------|
| `userService` | `src/services/userService.ts` | ✅ Compatible |
| `subscriptionService` | `src/services/subscriptionService.ts` | ✅ Compatible |
| `intakeService` | `src/services/intakeService.ts` | ✅ Compatible |
| `onboardingService` | `src/services/onboardingService.ts` | ✅ Compatible |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/reset-backend.sql` | Full database reset (preserves tiers) |
| `scripts/test-backend.mjs` | Integration test suite (20 tests) |

## Next Steps

1. **Create Emilio's account** via Supabase Dashboard
2. **Assign client role** and create client profile
3. **Test intake form flow** end-to-end
4. **Connect Stripe** for payment processing

## Environment Variables Required

```env
VITE_SUPABASE_URL=https://lvmcumsfpjjcgnnovvzs.supabase.co
VITE_SUPABASE_ANON_KEY=<your-anon-key>
STRIPE_SECRET_KEY=<your-stripe-key>
STRIPE_WEBHOOK_SECRET=<your-webhook-secret>
```
