# REPZ Database Migration Instructions

## Quick Method: Supabase Dashboard SQL Editor

1. **Open Supabase SQL Editor:**
   https://supabase.com/dashboard/project/lvmcumsfpjjcgnnovvzs/sql/new

2. **Copy the entire contents of this file:**
   `supabase/migrations/20251205_complete_intake_system.sql`

3. **Paste into the SQL Editor and click "Run"**

---

## Alternative: Supabase CLI

### Step 1: Login to Supabase
```bash
npx supabase login
```
This will open a browser window - complete the login.

### Step 2: Link the project
```bash
cd C:\Users\mesha\Desktop\GitHub\repz-llc\repz
npx supabase link --project-ref lvmcumsfpjjcgnnovvzs
```

### Step 3: Push migrations
```bash
npx supabase db push
```

---

## Alternative: Direct PostgreSQL Connection

### Step 1: Get your database password
Go to: https://supabase.com/dashboard/project/lvmcumsfpjjcgnnovvzs/settings/database
Copy the password from "Database password"

### Step 2: Run the migration script
```bash
set SUPABASE_DB_PASSWORD=your_password_here
npx tsx scripts/apply-migration.ts
```

---

## Verification Query

After applying the migration, run this query to verify:

```sql
-- Check tables created
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
  'intake_form_submissions',
  'intake_form_drafts', 
  'client_onboarding',
  'subscriptions',
  'coach_client_assignments',
  'activity_log'
);

-- Check functions created
SELECT routine_name 
FROM information_schema.routines 
WHERE routine_schema = 'public' 
AND routine_name IN (
  'update_onboarding_progress',
  'log_activity',
  'create_client_from_intake',
  'update_updated_at_column'
);
```

## Expected Results

### Tables (6 total):
- ✅ intake_form_submissions
- ✅ intake_form_drafts
- ✅ client_onboarding
- ✅ subscriptions
- ✅ coach_client_assignments
- ✅ activity_log

### Functions (4 total):
- ✅ update_onboarding_progress
- ✅ log_activity
- ✅ create_client_from_intake
- ✅ update_updated_at_column

---

## After Migration: Create Admin User

Run the seed script in SQL Editor:
`scripts/seed-users.sql`

This will set up:
- Admin account (you)
- Emilio's client account with Performance tier
