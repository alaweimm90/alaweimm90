import { chromium, FullConfig, Browser } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  console.log('🚀 Global Setup: Starting REPZ test environment');
  
  // Create browser instance for setup
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // Wait for dev server to be ready
    const baseURL = process.env.PLAYWRIGHT_TEST_BASE_URL || config.use?.baseURL || 'http://localhost:8080';
    console.log(`⏳ Waiting for dev server at ${baseURL}`);
    
    await page.goto(baseURL, { waitUntil: 'networkidle' });
    
    // Setup test database with seed data
    console.log('🌱 Seeding test database');
    await setupTestDatabase();
    
    // Create authenticated user sessions for different roles
    console.log('👤 Creating authenticated user sessions');
    await createAuthenticatedSessions(browser, baseURL);
    
    console.log('✅ Global Setup: Complete');
  } catch (error) {
    console.error('❌ Global Setup: Failed', error);
    throw error;
  } finally {
    await browser.close();
  }
}

async function setupTestDatabase() {
  // In a real scenario, you'd seed your test database here
  // For Supabase, you might run SQL scripts or use the JS client
  // Example:
  // await supabase.from('test_users').insert(testUsers);
  // await supabase.from('test_tiers').insert(testTiers);
  console.log('Database seeding completed (mock)');
}

async function createAuthenticatedSessions(browser: Browser, baseURL: string) {
  // Create different user session states
  const userTypes = [
    { role: 'client', tier: 'core', file: 'client-auth.json' },
    { role: 'client', tier: 'adaptive', file: 'adaptive-auth.json' },
    { role: 'client', tier: 'performance', file: 'performance-auth.json' },
    { role: 'client', tier: 'longevity', file: 'longevity-auth.json' },
    { role: 'coach', tier: null, file: 'coach-auth.json' },
    { role: 'admin', tier: null, file: 'admin-auth.json' },
    { role: 'medical', tier: null, file: 'medical-auth.json' },
  ];

  for (const userType of userTypes) {
    const context = await browser.newContext();
    const page = await context.newPage();
    
    try {
      await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
      try {
        const signInBtn = page.getByRole('button', { name: /sign in/i });
        await signInBtn.click();
        const continueBtn = page.getByRole('button', { name: /continue to sign in/i });
        await continueBtn.click();
      } catch {
        console.warn('')
      }
      await context.storageState({ path: `tests/fixtures/auth/${userType.file}` });
      console.log(`✅ Created auth session for ${userType.role} (${userType.tier || 'default'})`);
    } catch (error) {
      console.warn(`⚠️ Failed to create auth session for ${userType.role}:`, (error as Error).message);
    } finally {
      await context.close();
    }
  }
}

export default globalSetup;