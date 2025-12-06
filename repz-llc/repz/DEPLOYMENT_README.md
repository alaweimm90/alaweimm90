# REPZ Platform - Deployment Guide 🚀

## Project Overview

**REPZ Coach Pro** - A premium fitness coaching platform with:
- ✅ 7-step email intake form with tier selection
- ✅ 4 subscription tiers (Core $89 → Elite Concierge $349)
- ✅ PDF training plan generator
- ✅ Stripe payment integration
- ✅ Supabase backend configured
- ✅ Admin panel for client management

## Quick Start

```bash
# Navigate to platform directory
cd C:\Users\mesha\Desktop\REPZ-Platform\Repz\REPZ\platform

# Install dependencies (if needed)
npm install

# Run development server
npm run dev
```

**Live at:** http://localhost:8080

## Key Routes

| Route | Description |
|-------|-------------|
| `/` | Homepage |
| `/intake` | Landing page for intake form |
| `/intake-email` | 7-step intake form |
| `/intake-success` | Payment success page |
| `/pricing` | Pricing page |
| `/admin/non-portal-clients` | Admin: manage intake submissions |

## Tier Configuration

| Tier | Monthly Price | Annual Price | Key Features |
|------|---------------|--------------|--------------|
| **Core Program** | $89 | $71/mo | Foundation training, 72h response |
| **Adaptive Engine** | $149 | $119/mo | Biomarkers, wearables, 48h response |
| **Prime Suite** | $229 | $183/mo | AI assistant, PEDs protocols, 24h response |
| **Elite Concierge** | $349 | $279/mo | In-person 2x/week, 12h response |

## Environment Variables

Already configured in `.env`:
```env
VITE_SUPABASE_URL=https://lvmcumsfpjjcgnnovvzs.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGci...
```

## Database Setup

### Option 1: Via Supabase Dashboard
1. Go to https://supabase.com/dashboard/project/lvmcumsfpjjcgnnovvzs
2. Navigate to SQL Editor
3. Copy contents of `supabase/migrations/20250725000001_create_non_portal_clients.sql`
4. Run the SQL

### Option 2: Via Supabase CLI
```bash
# Login to Supabase
npx supabase login

# Link to project
npx supabase link --project-ref lvmcumsfpjjcgnnovvzs

# Push migrations
npx supabase db push
```

## Deploy to Vercel

### Step 1: Build the project
```bash
npm run build
```

### Step 2: Deploy
```bash
# Install Vercel CLI if needed
npm i -g vercel

# Deploy
vercel deploy --prod
```

### Step 3: Configure Environment Variables in Vercel
Add these env vars in Vercel Dashboard → Settings → Environment Variables:
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY`
- `STRIPE_SECRET_KEY` (when ready)
- `VITE_STRIPE_PUBLISHABLE_KEY` (when ready)

## Stripe Setup (When Ready)

1. Create Stripe products for each tier
2. Get Price IDs for monthly/annual billing
3. Add to environment variables:
   - `STRIPE_PRICE_CORE_MONTHLY`
   - `STRIPE_PRICE_CORE_ANNUAL`
   - etc.

## File Structure

```
platform/
├── src/
│   ├── pages/
│   │   ├── IntakeEmail.tsx      # 7-step intake form
│   │   ├── IntakeLanding.tsx    # Intake landing page
│   │   ├── IntakeSuccess.tsx    # Payment success
│   │   └── admin/
│   │       └── NonPortalClients.tsx  # Admin panel
│   ├── utils/
│   │   └── pdfGenerator.ts      # PDF plan generator
│   └── constants/
│       └── tiers.ts             # Tier configuration
├── supabase/
│   ├── functions/
│   │   └── create-intake-checkout/  # Stripe checkout
│   └── migrations/
│       └── 20250725000001_create_non_portal_clients.sql
└── .env                          # Environment config
```

## Support

For issues: Check the console logs and Supabase dashboard for debugging.

---
Built with ❤️ using React, Vite, Supabase, and Tailwind CSS
