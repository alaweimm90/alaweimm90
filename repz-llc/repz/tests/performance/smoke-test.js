import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 5 },   // Ramp up to 5 users
    { duration: '1m', target: 5 },    // Stay at 5 users
    { duration: '30s', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests under 2s
    http_req_failed: ['rate<0.1'],     // Error rate under 10%
    errors: ['rate<0.1'],              // Custom error rate under 10%
  },
};

// Test data
const BASE_URL = __ENV.REPZ_BASE_URL || 'http://localhost:8080';

const testUsers = [
  { email: 'test-core@repz.com', password: 'TestPassword123!', tier: 'core' },
  { email: 'test-adaptive@repz.com', password: 'TestPassword123!', tier: 'adaptive' },
  { email: 'test-performance@repz.com', password: 'TestPassword123!', tier: 'performance' },
];

export function setup() {
  console.log(`🚀 Starting REPZ smoke test against ${BASE_URL}`);
  
  // Health check
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (healthCheck.status !== 200) {
    console.error('❌ Health check failed, aborting test');
    return null;
  }
  
  console.log('✅ Health check passed');
  return { baseUrl: BASE_URL };
}

export default function (data) {
  if (!data) return;
  
  const baseUrl = data.baseUrl;
  const user = testUsers[Math.floor(Math.random() * testUsers.length)];
  
  // Test 1: Homepage load
  let response = http.get(baseUrl);
  let success = check(response, {
    'homepage loads successfully': (r) => r.status === 200,
    'homepage loads in reasonable time': (r) => r.timings.duration < 2000,
    'homepage contains expected content': (r) => r.body.includes('REPZ'),
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
  sleep(1);
  
  // Test 2: Authentication flow
  response = http.post(`${baseUrl}/api/auth/login`, JSON.stringify({
    email: user.email,
    password: user.password,
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  success = check(response, {
    'login request succeeds': (r) => r.status === 200 || r.status === 302,
    'login response time acceptable': (r) => r.timings.duration < 3000,
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
  
  // Extract auth token or session
  let authHeaders = {};
  if (response.headers['Set-Cookie']) {
    authHeaders['Cookie'] = response.headers['Set-Cookie'];
  }
  
  sleep(1);
  
  // Test 3: Dashboard load (authenticated)
  response = http.get(`${baseUrl}/dashboard`, { headers: authHeaders });
  success = check(response, {
    'dashboard loads successfully': (r) => r.status === 200,
    'dashboard loads quickly': (r) => r.timings.duration < 2500,
    'dashboard shows user content': (r) => r.body.includes('dashboard') || r.body.includes('Dashboard'),
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
  sleep(1);
  
  // Test 4: API endpoints based on tier
  if (user.tier === 'adaptive' || user.tier === 'performance') {
    // Test nutrition API
    response = http.get(`${baseUrl}/api/nutrition/food-search?q=chicken`, { headers: authHeaders });
    success = check(response, {
      'nutrition API responds': (r) => r.status === 200,
      'nutrition API is fast': (r) => r.timings.duration < 1500,
      'nutrition API returns data': (r) => r.json() && r.json().length > 0,
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
    sleep(1);
  }
  
  if (user.tier === 'performance') {
    // Test protocols API
    response = http.get(`${baseUrl}/api/protocols/available`, { headers: authHeaders });
    success = check(response, {
      'protocols API responds': (r) => r.status === 200,
      'protocols API is fast': (r) => r.timings.duration < 2000,
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
    sleep(1);
    
    // Test AI chat API
    response = http.post(`${baseUrl}/api/ai/chat`, JSON.stringify({
      message: 'What is optimal protein intake?',
      context: 'nutrition'
    }), {
      headers: { 
        ...authHeaders,
        'Content-Type': 'application/json' 
      },
    });
    
    success = check(response, {
      'AI API responds': (r) => r.status === 200,
      'AI API response time acceptable': (r) => r.timings.duration < 5000,
      'AI API returns response': (r) => r.json() && r.json().response,
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
    sleep(2);
  }
  
  // Test 5: Static asset loading
  const staticAssets = [
    '/assets/logo.svg',
    '/assets/main.css',
    '/assets/main.js',
  ];
  
  staticAssets.forEach(asset => {
    response = http.get(`${baseUrl}${asset}`);
    success = check(response, {
      [`${asset} loads successfully`]: (r) => r.status === 200 || r.status === 304,
      [`${asset} loads quickly`]: (r) => r.timings.duration < 1000,
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
  });
  
  sleep(1);
}

export function teardown(data) {
  console.log('🧹 Smoke test completed');
}