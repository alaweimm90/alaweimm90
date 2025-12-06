import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');
const requestCount = new Counter('total_requests');
const apiErrors = new Rate('api_errors');

// Stress test configuration - pushes system to breaking point
export const options = {
  stages: [
    { duration: '5m', target: 100 },   // Ramp up to 100 users
    { duration: '10m', target: 200 },  // Ramp up to 200 users
    { duration: '10m', target: 300 },  // Ramp up to 300 users
    { duration: '10m', target: 400 },  // Ramp up to 400 users (stress point)
    { duration: '5m', target: 500 },   // Push to 500 users (breaking point)
    { duration: '5m', target: 500 },   // Maintain breaking point
    { duration: '10m', target: 0 },    // Gradual recovery
  ],
  thresholds: {
    http_req_duration: [
      'p(50)<2000',   // 50% under 2s (degraded but acceptable)
      'p(95)<8000',   // 95% under 8s (stress conditions)
      'p(99)<15000',  // 99% under 15s (near failure)
    ],
    http_req_failed: ['rate<0.4'],      // Error rate under 40% (stress conditions)
    errors: ['rate<0.4'],               // Custom error rate under 40%
    api_errors: ['rate<0.3'],           // API error rate under 30%
    total_requests: ['count>5000'],     // At least 5000 requests
  },
};

const BASE_URL = __ENV.REPZ_BASE_URL || 'http://localhost:8080';

const testUsers = [
  { email: 'test-core@repz.com', password: 'TestPassword123!', tier: 'core' },
  { email: 'test-adaptive@repz.com', password: 'TestPassword123!', tier: 'adaptive' },
  { email: 'test-performance@repz.com', password: 'TestPassword123!', tier: 'performance' },
  { email: 'test-longevity@repz.com', password: 'TestPassword123!', tier: 'longevity' },
];

// Stress test scenarios - more aggressive than load test
const stressScenarios = {
  concurrent_api: 0.3,    // 30% users hitting APIs concurrently
  database_heavy: 0.4,    // 40% users doing DB-intensive operations
  mixed_chaos: 0.3,       // 30% users doing random mixed operations
};

export function setup() {
  console.log(`🔥 Starting REPZ stress test against ${BASE_URL}`);
  console.log(`⚠️  This test will push the system to its limits`);
  
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (healthCheck.status !== 200) {
    console.error('❌ Server not ready for stress test');
    return null;
  }
  
  return { baseUrl: BASE_URL };
}

export default function (data) {
  if (!data) return;
  
  const baseUrl = data.baseUrl;
  const user = testUsers[Math.floor(Math.random() * testUsers.length)];
  
  // Determine stress scenario
  const rand = Math.random();
  let scenario;
  if (rand < stressScenarios.concurrent_api) {
    scenario = 'concurrent_api';
  } else if (rand < stressScenarios.concurrent_api + stressScenarios.database_heavy) {
    scenario = 'database_heavy';
  } else {
    scenario = 'mixed_chaos';
  }
  
  // Execute stress scenario
  switch (scenario) {
    case 'concurrent_api':
      concurrentApiStress(baseUrl, user);
      break;
    case 'database_heavy':
      databaseHeavyStress(baseUrl, user);
      break;
    case 'mixed_chaos':
      mixedChaosStress(baseUrl, user);
      break;
  }
}

function concurrentApiStress(baseUrl, user) {
  // Hammers API endpoints with concurrent requests
  
  const authHeaders = authenticateUser(baseUrl, user);
  if (!authHeaders) return;
  
  // Rapid-fire API requests
  const apiEndpoints = [
    '/api/nutrition/food-search?q=protein',
    '/api/nutrition/food-search?q=carbs',
    '/api/nutrition/food-search?q=fats',
    '/api/user/profile',
    '/api/user/preferences',
    '/api/dashboard/stats',
  ];
  
  // Hit all endpoints rapidly
  apiEndpoints.forEach((endpoint, index) => {
    requestCount.add(1);
    
    const response = http.get(`${baseUrl}${endpoint}`, { 
      headers: authHeaders,
      timeout: '10s'  // Allow more time under stress
    });
    
    const success = check(response, {
      [`API ${index + 1} responds`]: (r) => r.status === 200 || r.status === 429, // Accept rate limiting
      [`API ${index + 1} not server error`]: (r) => r.status < 500,
    });
    
    errorRate.add(!success);
    apiErrors.add(response.status >= 400);
    responseTime.add(response.timings.duration);
    
    sleep(0.1); // Minimal sleep for stress
  });
  
  // Performance+ tier users stress AI endpoints
  if (user.tier === 'performance' || user.tier === 'longevity') {
    for (let i = 0; i < 5; i++) {
      requestCount.add(1);
      
      const response = http.post(`${baseUrl}/api/ai/chat`, JSON.stringify({
        message: `Stress test query ${i}`,
        context: 'performance'
      }), {
        headers: { 
          ...authHeaders,
          'Content-Type': 'application/json'
        },
        timeout: '30s'  // AI requests can take longer
      });
      
      const success = check(response, {
        'AI API handles stress': (r) => r.status === 200 || r.status === 429 || r.status === 503,
        'AI API not crashing': (r) => r.status !== 500,
      });
      
      errorRate.add(!success);
      apiErrors.add(response.status >= 400 && response.status !== 429);
      responseTime.add(response.timings.duration);
      
      sleep(0.2);
    }
  }
  
  sleep(1);
}

function databaseHeavyStress(baseUrl, user) {
  // Performs database-intensive operations
  
  const authHeaders = authenticateUser(baseUrl, user);
  if (!authHeaders) return;
  
  // Heavy search operations
  const complexSearches = [
    'chicken breast protein high quality organic',
    'quinoa gluten free ancient grain superfood',
    'avocado healthy fats monounsaturated omega',
    'salmon wild caught omega 3 fatty acids',
    'broccoli cruciferous vegetables fiber vitamins'
  ];
  
  complexSearches.forEach((query, index) => {
    requestCount.add(1);
    
    const response = http.get(`${baseUrl}/api/nutrition/food-search?q=${encodeURIComponent(query)}&limit=100&detailed=true`, {
      headers: authHeaders,
      timeout: '15s'
    });
    
    const success = check(response, {
      [`Complex search ${index + 1} works`]: (r) => r.status === 200 || r.status === 429,
      [`Complex search ${index + 1} not timeout`]: (r) => r.status !== 408,
    });
    
    errorRate.add(!success);
    apiErrors.add(response.status >= 400);
    responseTime.add(response.timings.duration);
    
    sleep(0.5);
  });
  
  // Recipe creation with complex calculations
  if (user.tier !== 'core') {
    requestCount.add(1);
    
    const complexRecipe = {
      name: `Stress Test Recipe ${Date.now()}`,
      ingredients: Array.from({ length: 20 }, (_, i) => ({
        foodId: `food-${i + 1}`,
        quantity: Math.floor(Math.random() * 500) + 50,
        unit: 'g'
      })),
      servings: Math.floor(Math.random() * 8) + 1
    };
    
    const response = http.post(`${baseUrl}/api/nutrition/recipes`, JSON.stringify(complexRecipe), {
      headers: { 
        ...authHeaders,
        'Content-Type': 'application/json'
      },
      timeout: '20s'
    });
    
    const success = check(response, {
      'Complex recipe creation handles stress': (r) => r.status === 200 || r.status === 201 || r.status === 429,
      'Recipe API not crashing': (r) => r.status < 500,
    });
    
    errorRate.add(!success);
    apiErrors.add(response.status >= 400 && response.status !== 429);
    responseTime.add(response.timings.duration);
    
    sleep(1);
  }
  
  // Protocol data queries for Performance+ users
  if (user.tier === 'performance' || user.tier === 'longevity') {
    const protocolQueries = [
      '/api/protocols/history',
      '/api/protocols/analytics',
      '/api/protocols/side-effects/all',
      '/api/medical/consultations/history',
      '/api/biomarkers/trends'
    ];
    
    protocolQueries.forEach((endpoint, index) => {
      requestCount.add(1);
      
      const response = http.get(`${baseUrl}${endpoint}?limit=1000&detailed=true`, {
        headers: authHeaders,
        timeout: '25s'
      });
      
      const success = check(response, {
        [`Protocol query ${index + 1} handles load`]: (r) => r.status === 200 || r.status === 429,
        [`Protocol query ${index + 1} not server error`]: (r) => r.status < 500,
      });
      
      errorRate.add(!success);
      apiErrors.add(response.status >= 400 && response.status !== 429);
      responseTime.add(response.timings.duration);
      
      sleep(0.3);
    });
  }
  
  sleep(2);
}

function mixedChaosStress(baseUrl, user) {
  // Random mix of operations to create unpredictable load
  
  const authHeaders = authenticateUser(baseUrl, user);
  if (!authHeaders) return;
  
  // Random page loads
  const pages = ['/dashboard', '/nutrition', '/progress', '/settings'];
  const randomPage = pages[Math.floor(Math.random() * pages.length)];
  
  requestCount.add(1);
  let response = http.get(`${baseUrl}${randomPage}`, { 
    headers: authHeaders,
    timeout: '10s'
  });
  
  let success = check(response, {
    'Random page loads under stress': (r) => r.status === 200 || r.status === 429,
    'Page not server error': (r) => r.status < 500,
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
  sleep(Math.random() * 2);
  
  // Random API operations
  const operations = [
    () => {
      // Rapid nutrition searches
      for (let i = 0; i < 3; i++) {
        requestCount.add(1);
        const query = ['protein', 'carbs', 'vitamins'][i];
        const response = http.get(`${baseUrl}/api/nutrition/food-search?q=${query}`, {
          headers: authHeaders,
          timeout: '8s'
        });
        
        const success = check(response, {
          [`Rapid search ${i + 1} works`]: (r) => r.status === 200 || r.status === 429,
        });
        
        errorRate.add(!success);
        apiErrors.add(response.status >= 400 && response.status !== 429);
        responseTime.add(response.timings.duration);
        sleep(0.1);
      }
    },
    
    () => {
      // Profile updates
      requestCount.add(1);
      const response = http.patch(`${baseUrl}/api/user/profile`, JSON.stringify({
        preferences: { theme: Math.random() > 0.5 ? 'dark' : 'light' }
      }), {
        headers: { 
          ...authHeaders,
          'Content-Type': 'application/json'
        },
        timeout: '10s'
      });
      
      const success = check(response, {
        'Profile update under stress': (r) => r.status === 200 || r.status === 429,
      });
      
      errorRate.add(!success);
      apiErrors.add(response.status >= 400 && response.status !== 429);
      responseTime.add(response.timings.duration);
    },
    
    () => {
      // File uploads simulation
      if (user.tier === 'performance' || user.tier === 'longevity') {
        requestCount.add(1);
        const response = http.post(`${baseUrl}/api/uploads/progress-photo`, {
          file: http.file('fake_data', 'fake-image.jpg', 'image/jpeg')
        }, {
          headers: authHeaders,
          timeout: '15s'
        });
        
        const success = check(response, {
          'File upload handles stress': (r) => r.status === 200 || r.status === 413 || r.status === 429,
        });
        
        errorRate.add(!success);
        apiErrors.add(response.status >= 500);
        responseTime.add(response.timings.duration);
      }
    }
  ];
  
  // Execute random operations
  const numOps = Math.floor(Math.random() * 3) + 1;
  for (let i = 0; i < numOps; i++) {
    const randomOp = operations[Math.floor(Math.random() * operations.length)];
    randomOp();
    sleep(Math.random() * 1);
  }
  
  sleep(Math.random() * 3);
}

function authenticateUser(baseUrl, user) {
  requestCount.add(1);
  
  const response = http.post(`${baseUrl}/api/auth/login`, JSON.stringify({
    email: user.email,
    password: user.password,
  }), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '10s'
  });
  
  const success = check(response, {
    'auth works under stress': (r) => r.status === 200 || r.status === 429,
    'auth not server error': (r) => r.status < 500,
  });
  
  errorRate.add(!success);
  apiErrors.add(response.status >= 400 && response.status !== 429);
  responseTime.add(response.timings.duration);
  
  if (!success) return null;
  
  let authHeaders = {};
  if (response.headers['Set-Cookie']) {
    authHeaders['Cookie'] = response.headers['Set-Cookie'];
  }
  if (response.json && response.json().token) {
    authHeaders['Authorization'] = `Bearer ${response.json().token}`;
  }
  
  return authHeaders;
}

export function teardown(data) {
  console.log('🔥 Stress test completed');
  console.log('📊 System performance under extreme load documented');
  console.log('🔍 Review breaking points and recovery patterns');
}