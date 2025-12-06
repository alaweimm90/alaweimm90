/**
 * Apply Database Migration via Supabase REST API
 *
 * Uses the Supabase service role key to execute SQL via the REST API
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SUPABASE_URL = 'https://lvmcumsfpjjcgnnovvzs.supabase.co';
const SUPABASE_SERVICE_ROLE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx2bWN1bXNmcGpqY2dubm92dnpzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzQzODE2MywiZXhwIjoyMDY5MDE0MTYzfQ.hfbXKKaDhyTOi51SQw3eITkv2zdOx3c9ZkdkG_UX5kM';

/**
 * Execute SQL via Supabase's pg_graphql or direct query
 */
async function executeSQLViaAPI(sql) {
  // Try using the SQL endpoint (if available)
  const response = await fetch(`${SUPABASE_URL}/rest/v1/rpc/exec_sql`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'apikey': SUPABASE_SERVICE_ROLE_KEY,
      'Authorization': `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
      'Prefer': 'return=representation'
    },
    body: JSON.stringify({ query: sql })
  });

  return response;
}

/**
 * Create the exec_sql function if it doesn't exist
 */
async function createExecSQLFunction() {
  const createFunctionSQL = `
    CREATE OR REPLACE FUNCTION exec_sql(query text)
    RETURNS json
    LANGUAGE plpgsql
    SECURITY DEFINER
    AS $$
    DECLARE
      result json;
    BEGIN
      EXECUTE query;
      RETURN json_build_object('success', true);
    EXCEPTION WHEN OTHERS THEN
      RETURN json_build_object('success', false, 'error', SQLERRM);
    END;
    $$;
  `;

  // This won't work via REST API without the function existing first
  // We need to use a different approach
  return false;
}

/**
 * Split SQL into executable statements
 */
function splitSQLStatements(sql) {
  const statements = [];
  let current = '';
  let inDollarQuote = false;
  let dollarTag = '';

  const lines = sql.split('\n');

  for (const line of lines) {
    const trimmed = line.trim();

    // Skip pure comment lines at the start of a statement
    if (trimmed.startsWith('--') && !current.trim()) {
      continue;
    }

    // Check for dollar quotes
    const dollarMatches = line.match(/\$[a-zA-Z_]*\$/g);
    if (dollarMatches) {
      for (const match of dollarMatches) {
        if (!inDollarQuote) {
          inDollarQuote = true;
          dollarTag = match;
        } else if (match === dollarTag) {
          inDollarQuote = false;
          dollarTag = '';
        }
      }
    }

    current += line + '\n';

    // End of statement
    if (!inDollarQuote && trimmed.endsWith(';')) {
      const stmt = current.trim();
      if (stmt && !stmt.match(/^--/)) {
        statements.push(stmt);
      }
      current = '';
    }
  }

  if (current.trim()) {
    statements.push(current.trim());
  }

  return statements;
}

async function applyMigration() {
  console.log('🚀 REPZ Database Migration Tool (API Mode)\n');
  console.log('📍 Project: lvmcumsfpjjcgnnovvzs');
  console.log('📍 URL:', SUPABASE_URL, '\n');

  // Read migration file
  const migrationPath = path.join(__dirname, '../supabase/migrations/20251205_complete_intake_system.sql');
  const migrationSQL = fs.readFileSync(migrationPath, 'utf-8');

  console.log('📄 Migration file loaded');
  console.log(`   Size: ${migrationSQL.length} characters\n`);

  // Test connection by querying existing tables
  console.log('🔌 Testing connection...');

  const testResponse = await fetch(`${SUPABASE_URL}/rest/v1/`, {
    method: 'GET',
    headers: {
      'apikey': SUPABASE_SERVICE_ROLE_KEY,
      'Authorization': `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`
    }
  });

  if (!testResponse.ok) {
    console.error('❌ Connection failed:', testResponse.status, testResponse.statusText);
    process.exit(1);
  }

  const tables = await testResponse.json();
  console.log('✅ Connected! Found', Object.keys(tables).length, 'existing endpoints\n');

  // Check if tables already exist
  const existingTables = Object.keys(tables);
  const targetTables = ['intake_form_submissions', 'intake_form_drafts', 'client_onboarding', 'subscriptions', 'coach_client_assignments', 'activity_log'];

  const alreadyExist = targetTables.filter(t => existingTables.includes(t));
  if (alreadyExist.length > 0) {
    console.log('⚠️  Some tables already exist:', alreadyExist.join(', '));
    console.log('   Migration will recreate them.\n');
  }

  // Since we can't execute raw SQL via REST API without a function,
  // we need to output instructions
  console.log('═'.repeat(60));
  console.log('📋 MANUAL MIGRATION REQUIRED');
  console.log('═'.repeat(60));
  console.log('\nThe Supabase REST API does not support raw SQL execution.');
  console.log('Please apply the migration using one of these methods:\n');

  console.log('Option 1: Supabase Dashboard (Recommended)');
  console.log('─'.repeat(40));
  console.log('1. Go to: https://supabase.com/dashboard/project/lvmcumsfpjjcgnnovvzs/sql');
  console.log('2. Copy the contents of:');
  console.log('   supabase/migrations/20251205_complete_intake_system.sql');
  console.log('3. Paste into the SQL Editor and click "Run"\n');

  console.log('Option 2: Supabase CLI');
  console.log('─'.repeat(40));
  console.log('1. Run: npx supabase link --project-ref lvmcumsfpjjcgnnovvzs');
  console.log('2. Run: npx supabase db push\n');

  console.log('Option 3: Direct PostgreSQL Connection');
  console.log('─'.repeat(40));
  console.log('1. Get your database password from Supabase Dashboard > Settings > Database');
  console.log('2. Set environment variable: SUPABASE_DB_PASSWORD=your_password');
  console.log('3. Run: npx tsx scripts/apply-migration.ts\n');

  // Output the SQL for easy copy
  console.log('═'.repeat(60));
  console.log('📄 MIGRATION SQL (copy this to Supabase SQL Editor):');
  console.log('═'.repeat(60));
  console.log('\n' + migrationSQL.substring(0, 500) + '...\n');
  console.log(`[Full SQL: ${migrationSQL.length} characters - see migration file]`);

  // Write a simplified version for quick testing
  const quickTestSQL = `
-- Quick verification query
SELECT
  (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public') as table_count,
  (SELECT COUNT(*) FROM information_schema.routines WHERE routine_schema = 'public') as function_count;
  `;

  console.log('\n📊 To verify after migration, run this query:');
  console.log(quickTestSQL);
}

applyMigration().catch(console.error);
