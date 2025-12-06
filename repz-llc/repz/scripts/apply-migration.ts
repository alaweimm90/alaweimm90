/**
 * Apply Database Migration to Supabase
 *
 * This script connects directly to Supabase PostgreSQL and executes the migration SQL
 */

import { Client } from 'pg';
import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';

// Supabase connection details
const PROJECT_REF = 'lvmcumsfpjjcgnnovvzs';

async function getPassword(): Promise<string> {
  // Try to get from environment first
  if (process.env.SUPABASE_DB_PASSWORD) {
    return process.env.SUPABASE_DB_PASSWORD;
  }

  // Prompt for password
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  return new Promise((resolve) => {
    rl.question('Enter your Supabase database password: ', (answer) => {
      rl.close();
      resolve(answer);
    });
  });
}

async function applyMigration() {
  console.log('🚀 REPZ Database Migration Tool\n');
  console.log('📍 Project: lvmcumsfpjjcgnnovvzs');
  console.log('📍 Host: db.lvmcumsfpjjcgnnovvzs.supabase.co\n');

  const password = await getPassword();

  const client = new Client({
    host: `db.${PROJECT_REF}.supabase.co`,
    port: 5432,
    database: 'postgres',
    user: 'postgres',
    password: password,
    ssl: { rejectUnauthorized: false }
  });

  try {
    console.log('\n🔌 Connecting to database...');
    await client.connect();
    console.log('✅ Connected successfully!\n');

    // Read migration file
    const migrationPath = path.join(__dirname, '../supabase/migrations/20251205_complete_intake_system.sql');
    const migrationSQL = fs.readFileSync(migrationPath, 'utf-8');

    console.log('📄 Migration file loaded');
    console.log(`   Size: ${migrationSQL.length} characters\n`);

    // Execute the entire migration as a transaction
    console.log('🔄 Executing migration...\n');

    await client.query('BEGIN');

    try {
      await client.query(migrationSQL);
      await client.query('COMMIT');
      console.log('✅ Migration executed successfully!\n');
    } catch (err: any) {
      await client.query('ROLLBACK');
      console.error('❌ Migration failed:', err.message);
      console.error('\nRolled back all changes.');
      throw err;
    }

    // Verify tables were created
    console.log('🔍 Verifying created tables...\n');

    const tablesResult = await client.query(`
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
      )
      ORDER BY table_name
    `);

    console.log('📊 Tables verified:');
    for (const row of tablesResult.rows) {
      console.log(`   ✅ ${row.table_name}`);
    }

    // Check functions
    const functionsResult = await client.query(`
      SELECT routine_name
      FROM information_schema.routines
      WHERE routine_schema = 'public'
      AND routine_name IN (
        'update_onboarding_progress',
        'log_activity',
        'create_client_from_intake',
        'update_updated_at_column'
      )
      ORDER BY routine_name
    `);

    console.log('\n📊 Functions verified:');
    for (const row of functionsResult.rows) {
      console.log(`   ✅ ${row.routine_name}`);
    }

    console.log('\n🎉 Migration complete! All tables and functions created successfully.');

  } catch (err: any) {
    console.error('\n❌ Error:', err.message);
    process.exit(1);
  } finally {
    await client.end();
    console.log('\n🔌 Database connection closed.');
  }
}

applyMigration().catch(console.error);
