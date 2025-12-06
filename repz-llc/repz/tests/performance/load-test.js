import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');
const requestCount = new Counter('total_requests');

// Load test configuration
export const options = {
  stages: [
    { duration: '2m', target: 20 },   // Ramp up to 20 users over 2 minutes
    { duration: '5m', target: 20 },   // Stay at 20 users for 5 minutes
    { duration: '2m', target: 50 },   // Ramp up to 50 users over 2 minutes
    { duration: '5m', target: 50 },   // Stay at 50 users for 5 minutes
    { duration: '2m', target: 100 },  // Ramp up to 100 users over 2 minutes
    { duration: '5m', target: 100 },  // Stay at 100 users for 5 minutes
    { duration: '3m', target: 0 },    // Ramp down to 0 users over 3 minutes
  ],
  thresholds: {
    http_req_duration: [
      'p(50)<1000',  // 50% of requests under 1s
      'p(95)<2500',  // 95% of requests under 2.5s
      'p(99)<5000',  // 99% of requests under 5s
    ],
    http_req_failed: ['rate<0.05'],     // Error rate under 5%
    errors: ['rate<0.05'],              // Custom error rate under 5%
    total_requests: ['count>1000'],     // At least 1000 requests total
  },
};

// Test configuration
const BASE_URL = __ENV.REPZ_BASE_URL || 'http://localhost:8080';

const testUsers = [
  { email: 'test-core@repz.com', password: 'TestPassword123!', tier: 'core' },
  { email: 'test-adaptive@repz.com', password: 'TestPassword123!', tier: 'adaptive' },
  { email: 'test-performance@repz.com', password: 'TestPassword123!', tier: 'performance' },
  { email: 'test-longevity@repz.com', password: 'TestPassword123!', tier: 'longevity' },
];

// User behavior patterns
const userScenarios = {
  browsing: 0.3,      // 30% browsing users
  active: 0.5,        // 50% active users
  power: 0.2,         // 20% power users
};

export function setup() {
  console.log(`🚀 Starting REPZ load test against ${BASE_URL}`);
  console.log(`📊 Test will simulate realistic user behavior patterns`);
  
  // Verify server is ready
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (healthCheck.status !== 200) {
    console.error('❌ Server health check failed');
    return null;
  }
  
  return { baseUrl: BASE_URL };
}

export default function (data) {
  if (!data) return;
  
  const baseUrl = data.baseUrl;
  const user = testUsers[Math.floor(Math.random() * testUsers.length)];
  
  // Determine user behavior type
  const rand = Math.random();
  let behaviorType;
  if (rand < userScenarios.browsing) {
    behaviorType = 'browsing';
  } else if (rand < userScenarios.browsing + userScenarios.active) {
    behaviorType = 'active';
  } else {
    behaviorType = 'power';
  }
  
  // Execute scenario based on behavior type
  switch (behaviorType) {
    case 'browsing':
      browsingUserScenario(baseUrl, user);
      break;
    case 'active':
      activeUserScenario(baseUrl, user);
      break;
    case 'power':
      powerUserScenario(baseUrl, user);
      break;
  }
}

function browsingUserScenario(baseUrl, user) {
  // Simulates casual browsing behavior
  requestCount.add(1);
  
  // Visit homepage
  let response = http.get(baseUrl);
  let success = check(response, {
    'homepage loads': (r) => r.status === 200,
    'homepage performance': (r) => r.timings.duration < 2000,
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
  sleep(2);
  
  // Browse pricing
  requestCount.add(1);
  response = http.get(`${baseUrl}/pricing`);
  success = check(response, {
    'pricing page loads': (r) => r.status === 200,
    'pricing performance': (r) => r.timings.duration < 1500,
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
  sleep(3);
  
  // Maybe sign up (20% chance)
  if (Math.random() < 0.2) {
    response = http.get(`${baseUrl}/auth/signup`);
    success = check(response, {
      'signup page loads': (r) => r.status === 200,
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
  }
  
  sleep(5); // Browsing pause
}

function activeUserScenario(baseUrl, user) {
  // Simulates regular user activity
  
  // Authenticate
  const authHeaders = authenticateUser(baseUrl, user);
  if (!authHeaders) return;
  
  sleep(1);
  
  // Visit dashboard
  requestCount.add(1);
  let response = http.get(`${baseUrl}/dashboard`, { headers: authHeaders });
  let success = check(response, {
    'dashboard loads': (r) => r.status === 200,
    'dashboard performance': (r) => r.timings.duration < 2000,
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
  sleep(2);
  
  // Activity based on tier
  if (user.tier === 'adaptive' || user.tier === 'performance' || user.tier === 'longevity') {
    // Use nutrition features
    requestCount.add(1);
    response = http.get(`${baseUrl}/nutrition/food-database`, { headers: authHeaders });
    success = check(response, {
      'nutrition page loads': (r) => r.status === 200,
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
    sleep(1);
    
    // Search food database
    requestCount.add(1);
    response = http.get(`${baseUrl}/api/nutrition/food-search?q=chicken&limit=20`, { headers: authHeaders });
    success = check(response, {
      'food search API works': (r) => r.status === 200,
      'food search is fast': (r) => r.timings.duration < 1500,
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
    sleep(2);
    
    // View meal planning
    if (Math.random() < 0.6) {
      requestCount.add(1);
      response = http.get(`${baseUrl}/nutrition/meal-planning`, { headers: authHeaders });
      success = check(response, {
        'meal planning loads': (r) => r.status === 200,
      });
      
      errorRate.add(!success);
      responseTime.add(response.timings.duration);
      sleep(3);
    }
  }
  
  // Check progress/analytics
  if (Math.random() < 0.4) {
    requestCount.add(1);
    response = http.get(`${baseUrl}/progress`, { headers: authHeaders });
    success = check(response, {
      'progress page loads': (r) => r.status === 200,
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
    sleep(2);
  }
  
  sleep(3);
}

function powerUserScenario(baseUrl, user) {
  // Simulates heavy usage by advanced users
  
  // Authenticate
  const authHeaders = authenticateUser(baseUrl, user);
  if (!authHeaders) return;
  
  sleep(0.5);
  
  // Dashboard
  requestCount.add(1);
  let response = http.get(`${baseUrl}/dashboard`, { headers: authHeaders });
  let success = check(response, {
    'dashboard loads': (r) => r.status === 200,
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
  sleep(1);
  
  // Heavy API usage
  if (user.tier === 'performance' || user.tier === 'longevity') {
    // Multiple protocol requests
    const protocols = ['testosterone', 'growth-hormone', 'bpc157'];
    protocols.forEach(protocol => {
      requestCount.add(1);
      response = http.get(`${baseUrl}/api/protocols/${protocol}`, { headers: authHeaders });
      success = check(response, {
        [`${protocol} API works`]: (r) => r.status === 200,
      });
      
      errorRate.add(!success);
      responseTime.add(response.timings.duration);
      sleep(0.5);
    });
    
    // AI interactions
    for (let i = 0; i < 3; i++) {
      requestCount.add(1);
      response = http.post(`${baseUrl}/api/ai/chat`, JSON.stringify({
        message: `AI query ${i + 1}`,
        context: 'training'
      }), {
        headers: { 
          ...authHeaders,
          'Content-Type': 'application/json'
        },
      });
      
      success = check(response, {
        'AI API responds': (r) => r.status === 200,
        'AI response time reasonable': (r) => r.timings.duration < 8000,
      });
      
      errorRate.add(!success);
      responseTime.add(response.timings.duration);
      sleep(1);
    }
    
    // Form analysis upload simulation
    if (Math.random() < 0.3) {
      requestCount.add(1);
      response = http.post(`${baseUrl}/api/ai/form-analysis`, JSON.stringify({
        exercise: 'squat',
        videoData: 'base64_mock_data'
      }), {
        headers: { 
          ...authHeaders,
          'Content-Type': 'application/json'
        },
      });
      
      success = check(response, {
        'form analysis API works': (r) => r.status === 200 || r.status === 202,
        'form analysis accepts request': (r) => r.timings.duration < 3000,
      });
      
      errorRate.add(!success);
      responseTime.add(response.timings.duration);
      sleep(2);
    }
  }
  
  // Extensive nutrition usage
  if (user.tier !== 'core') {
    // Multiple food searches
    const searches = ['chicken', 'quinoa', 'avocado', 'salmon', 'broccoli'];
    searches.forEach(query => {
      requestCount.add(1);
      response = http.get(`${baseUrl}/api/nutrition/food-search?q=${query}&limit=50`, { headers: authHeaders });
      success = check(response, {
        [`${query} search works`]: (r) => r.status === 200,
        [`${query} search is fast`]: (r) => r.timings.duration < 1200,
      });
      
      errorRate.add(!success);
      responseTime.add(response.timings.duration);
      sleep(0.3);
    });
    
    // Recipe creation
    if (Math.random() < 0.4) {
      requestCount.add(1);
      response = http.post(`${baseUrl}/api/nutrition/recipes`, JSON.stringify({
        name: 'Power Bowl Test',
        ingredients: [
          { foodId: 'chicken-breast', quantity: 200 },
          { foodId: 'quinoa', quantity: 100 }
        ]
      }), {
        headers: { 
          ...authHeaders,
          'Content-Type': 'application/json'
        },
      });
      
      success = check(response, {
        'recipe creation works': (r) => r.status === 200 || r.status === 201,
      });
      
      errorRate.add(!success);
      responseTime.add(response.timings.duration);
      sleep(1);
    }
    
    // Meal plan operations
    requestCount.add(1);
    response = http.get(`${baseUrl}/api/nutrition/meal-plan/current`, { headers: authHeaders });
    success = check(response, {
      'meal plan API works': (r) => r.status === 200,
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
    sleep(1);
  }
  
  sleep(2);
}

function authenticateUser(baseUrl, user) {
  requestCount.add(1);
  
  const response = http.post(`${baseUrl}/api/auth/login`, JSON.stringify({
    email: user.email,
    password: user.password,
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  const success = check(response, {
    'authentication succeeds': (r) => r.status === 200 || r.status === 302,
    'auth response time acceptable': (r) => r.timings.duration < 3000,
  });
  
  errorRate.add(!success);
  responseTime.add(response.timings.duration);
  
  if (!success) return null;
  
  // Extract auth headers
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
  console.log('🏁 Load test completed');
  console.log('📈 Check results for performance insights');
}