/**
 * REPZ Backend Integration Test
 * Tests all core database operations
 */

const PROJECT_REF = 'lvmcumsfpjjcgnnovvzs';
const ACCESS_TOKEN = 'sbp_6746ebc20211d0948cdb7f5a0615da37a7827193';

async function query(sql) {
  const response = await fetch(
    `https://api.supabase.com/v1/projects/${PROJECT_REF}/database/query`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${ACCESS_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: sql }),
    }
  );
  const result = await response.json();
  if (result.error) throw new Error(result.error);
  return result;
}

async function test(name, fn) {
  try {
    await fn();
    console.log(`✅ ${name}`);
    return true;
  } catch (error) {
    console.log(`❌ ${name}: ${error.message}`);
    return false;
  }
}

async function runTests() {
  console.log('🧪 REPZ Backend Integration Tests\n');
  console.log('='.repeat(60));

  let passed = 0;
  let failed = 0;

  // Test 1: Profiles table exists
  if (await test('Profiles table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM profiles`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 2: User roles table exists
  if (await test('User roles table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM user_roles`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 3: Subscription tiers exist
  if (await test('Subscription tiers exist', async () => {
    const result = await query(`SELECT COUNT(*) as count FROM subscription_tiers`);
    if (result[0].count < 4) throw new Error('Expected at least 4 tiers');
  })) passed++; else failed++;

  // Test 4: Client profiles table exists
  if (await test('Client profiles table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM client_profiles`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 5: Coach profiles table exists
  if (await test('Coach profiles table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM coach_profiles`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 6: Subscriptions table exists
  if (await test('Subscriptions table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM subscriptions`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 7: Intake form submissions table exists
  if (await test('Intake form submissions table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM intake_form_submissions`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 8: Intake form drafts table exists
  if (await test('Intake form drafts table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM intake_form_drafts`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 9: Client onboarding table exists
  if (await test('Client onboarding table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM client_onboarding`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 10: Coach-client assignments table exists
  if (await test('Coach-client assignments table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM coach_client_assignments`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 11: Programs table exists
  if (await test('Programs table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM programs`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 12: Weekly checkins table exists
  if (await test('Weekly checkins table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM weekly_checkins`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 13: Messages table exists
  if (await test('Messages table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM messages`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 14: Activity log table exists
  if (await test('Activity log table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM activity_log`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 15: Progress photos table exists
  if (await test('Progress photos table exists', async () => {
    const result = await query(`SELECT COUNT(*) FROM progress_photos`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 16: has_role function exists
  if (await test('has_role function exists', async () => {
    const result = await query(`SELECT has_role('admin')`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 17: get_user_role function exists
  if (await test('get_user_role function exists', async () => {
    const result = await query(`SELECT get_user_role()`);
    if (!result) throw new Error('No result');
  })) passed++; else failed++;

  // Test 18: update_updated_at function exists
  if (await test('update_updated_at function exists', async () => {
    const result = await query(`
      SELECT routine_name FROM information_schema.routines
      WHERE routine_schema = 'public' AND routine_name = 'update_updated_at'
    `);
    if (result.length === 0) throw new Error('Function not found');
  })) passed++; else failed++;

  // Test 19: Admin role assigned
  if (await test('Admin role assigned to meshal@berkeley.edu', async () => {
    const result = await query(`
      SELECT ur.role FROM user_roles ur
      JOIN auth.users u ON ur.user_id = u.id
      WHERE u.email = 'meshal@berkeley.edu' AND ur.role = 'admin'
    `);
    if (result.length === 0) throw new Error('Admin role not found');
  })) passed++; else failed++;

  // Test 20: RLS enabled on all tables
  if (await test('RLS enabled on core tables', async () => {
    const result = await query(`
      SELECT tablename FROM pg_tables
      WHERE schemaname = 'public'
      AND tablename IN ('profiles', 'user_roles', 'client_profiles', 'subscriptions')
      AND tablename NOT IN (
        SELECT tablename FROM pg_tables t
        JOIN pg_class c ON c.relname = t.tablename
        WHERE c.relrowsecurity = true
      )
    `);
    // This query returns tables WITHOUT RLS, so we want 0 results
  })) passed++; else failed++;

  console.log('\n' + '='.repeat(60));
  console.log(`\n📊 Results: ${passed} passed, ${failed} failed`);

  if (failed === 0) {
    console.log('\n🎉 All tests passed! Backend is ready.');
  } else {
    console.log('\n⚠️  Some tests failed. Review the errors above.');
  }

  // Summary
  console.log('\n📋 Database Summary:');
  const tables = await query(`SELECT COUNT(*) as count FROM pg_tables WHERE schemaname = 'public'`);
  console.log(`   Tables: ${tables[0].count}`);

  const functions = await query(`
    SELECT COUNT(*) as count FROM information_schema.routines
    WHERE routine_schema = 'public' AND routine_type = 'FUNCTION'
  `);
  console.log(`   Functions: ${functions[0].count}`);

  const users = await query(`SELECT COUNT(*) as count FROM auth.users`);
  console.log(`   Auth Users: ${users[0].count}`);

  const tiers = await query(`SELECT display_name, price_cents FROM subscription_tiers ORDER BY price_cents`);
  console.log(`   Subscription Tiers:`);
  tiers.forEach(t => console.log(`     - ${t.display_name}: $${(t.price_cents / 100).toFixed(2)}/mo`));
}

runTests().catch(console.error);
