#!/usr/bin/env tsx
/**
 * Health check for all services
 */
const SERVICES = [
  { name: 'AI Proxy', url: 'http://localhost:4000/health' },
  { name: 'REPZ API', url: 'http://localhost:3000/api/health' },
];

async function checkHealth() {
  console.log('🏥 Health Check\n');
  for (const svc of SERVICES) {
    try {
      const res = await fetch(svc.url, { signal: AbortSignal.timeout(2000) });
      console.log(`  ${svc.name}: ${res.ok ? '✅ OK' : '❌ DOWN'}`);
    } catch {
      console.log(`  ${svc.name}: ⚪ Not running`);
    }
  }
}

checkHealth();
