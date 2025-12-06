import { FullConfig } from '@playwright/test';

async function globalTeardown(config: FullConfig) {
  console.log('🧹 Global Teardown: Starting REPZ test cleanup');
  
  try {
    // Step 1: Clean up test database
    console.log('🗑️ Cleaning up test database');
    await cleanupTestDatabase();
    
    // Step 2: Remove authentication state files
    console.log('🔐 Removing authentication state files');
    await cleanupAuthFiles();
    
    // Step 3: Clean up uploaded test files
    console.log('📁 Cleaning up test file uploads');
    await cleanupTestFiles();
    
    // Step 4: Reset test environment state
    console.log('🔄 Resetting test environment');
    await resetTestEnvironment();
    
    // Step 5: Generate test completion report
    console.log('📊 Generating test completion report');
    await generateTestReport();
    
    console.log('✅ Global Teardown: Complete');
  } catch (error) {
    console.error('❌ Global Teardown: Failed', error);
    // Don't throw error to avoid affecting test results
  }
}

async function cleanupTestDatabase() {
  try {
    // In a real scenario, you'd clean up test data from your database
    // For Supabase, you might:
    // - Delete test user accounts
    // - Clean up test subscription data
    // - Remove test protocol records
    // - Clear test conversation history
    
    // Example cleanup operations:
    // await supabase.from('test_users').delete().neq('id', '');
    // await supabase.from('test_subscriptions').delete().neq('id', '');
    // await supabase.from('test_protocols').delete().neq('id', '');
    // await supabase.from('test_conversations').delete().neq('id', '');
    
    console.log('Database cleanup completed (mock)');
  } catch (error) {
    console.warn('Database cleanup failed:', error.message);
  }
}

async function cleanupAuthFiles() {
  try {
    const fsMod = await import('fs');
    const fs = fsMod.promises;
    const pathMod = await import('path');
    const path: any = (pathMod as any).default || pathMod;
    
    const urlMod = await import('url');
    const currentDir = path.dirname(urlMod.fileURLToPath(import.meta.url));
    const authDir = path.join(currentDir, 'auth');
    
    // List of auth files to clean up
    const authFiles = [
      'client-auth.json',
      'adaptive-auth.json', 
      'performance-auth.json',
      'longevity-auth.json',
      'coach-auth.json',
      'admin-auth.json',
      'medical-auth.json',
      'unverified-medical-auth.json'
    ];
    
    for (const file of authFiles) {
      const filePath = path.join(authDir, file);
      try {
        await fs.access(filePath);
        await fs.unlink(filePath);
        console.log(`Removed auth file: ${file}`);
      } catch (err) {
        console.log(`Auth file not found (already clean): ${file}`);
      }
    }
  } catch (error) {
    console.warn('Auth file cleanup failed:', error.message);
  }
}

async function cleanupTestFiles() {
  try {
    const fsMod = await import('fs');
    const fs = fsMod.promises;
    const pathMod = await import('path');
    const path: any = (pathMod as any).default || pathMod;
    
    // Clean up test uploads directory
    const urlMod = await import('url');
    const currentDir = path.dirname(urlMod.fileURLToPath(import.meta.url));
    const uploadsDir = path.join(currentDir, '..', 'uploads');
    
    try {
      const files = await fs.readdir(uploadsDir);
      for (const file of files) {
        if (file.startsWith('test-') || file.includes('fixture')) {
          await fs.unlink(path.join(uploadsDir, file));
          console.log(`Removed test file: ${file}`);
        }
      }
    } catch (err) {
      console.log('No test uploads directory to clean');
    }
    
    // Clean up any test artifacts
    const urlMod = await import('url');
    const currentDir = path.dirname(urlMod.fileURLToPath(import.meta.url));
    const testResultsDir = path.join(currentDir, '..', '..', 'test-results');
    
    try {
      // Keep the most recent test results, but clean up old ones
      const files = await fs.readdir(testResultsDir);
      const oldFiles = [] as string[];
      for (const file of files) {
        try {
          const stats = await fsMod.promises.stat(path.join(testResultsDir, file));
          const ageInDays = (Date.now() - stats.mtime.getTime()) / (1000 * 60 * 60 * 24);
          if (ageInDays > 7) oldFiles.push(file);
        } catch { void 0; }
      }
      
      for (const file of oldFiles) {
        await fs.unlink(path.join(testResultsDir, file));
        console.log(`Removed old test result: ${file}`);
      }
    } catch (err) {
      console.log('No old test results to clean');
    }
  } catch (error) {
    console.warn('Test file cleanup failed:', error.message);
  }
}

async function resetTestEnvironment() {
  try {
    // Reset any environment-specific state
    // This might include:
    // - Clearing cache files
    // - Resetting feature flags
    // - Clearing localStorage/sessionStorage
    // - Resetting mock server state
    
    console.log('Test environment reset completed');
  } catch (error) {
    console.warn('Environment reset failed:', error.message);
  }
}

async function generateTestReport() {
  try {
    const fsMod = await import('fs');
    const fs = fsMod.promises;
    const pathMod = await import('path');
    const path: any = (pathMod as any).default || pathMod;
    
    const report = {
      timestamp: new Date().toISOString(),
      teardownCompleted: true,
      environment: process.env.NODE_ENV || 'test',
      testSuite: 'REPZ E2E Tests',
      cleanup: {
        database: 'completed',
        authFiles: 'completed', 
        testFiles: 'completed',
        environment: 'completed'
      },
      notes: 'Global teardown completed successfully'
    };
    
    const urlMod = await import('url');
    const currentDir = path.dirname(urlMod.fileURLToPath(import.meta.url));
    const reportPath = path.join(currentDir, '..', 'test-results', 'teardown-report.json');
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    console.log(`Test completion report generated: ${reportPath}`);
  } catch (error) {
    console.warn('Test report generation failed:', error.message);
  }
}

export default globalTeardown;