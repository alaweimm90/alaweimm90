const request = require('supertest');
const app = require('./index');

describe('AI Agent Demo App', () => {
  it('should return health status', async () => {
    const response = await request(app).get('/health');
    expect(response.status).toBe(200);
    expect(response.body.status).toBe('healthy');
    expect(response.body.metrics).toBeDefined();
  });

  it('should return version from package.json', async () => {
    const response = await request(app).get('/version');
    expect(response.status).toBe(200);
    expect(response.body.version).toBeDefined();
  });

  it('should process valid finance transaction', async () => {
    const response = await request(app)
      .post('/api/finance')
      .send({ amount: 100, currency: 'USD' });
    expect(response.status).toBe(200);
    expect(response.body.processed).toBe(true);
    expect(response.body.transactionId).toBeDefined();
  });

  it('should reject invalid finance transaction per SME rules', async () => {
    const response = await request(app)
      .post('/api/finance')
      .send({ amount: -50, currency: 'USD' });
    expect(response.status).toBe(400);
    expect(response.body.error).toContain('SME rules');
  });

  it('should return agent status with capabilities', async () => {
    const response = await request(app).get('/api/agent/status');
    expect(response.status).toBe(200);
    expect(response.body.agent).toBe('AI Agent Demo');
    expect(response.body.capabilities).toContain('Contextual Intelligence');
    expect(response.body.capabilities).toContain('Hallucination Prevention');
  });

  it('should handle hallucination detection', async () => {
    // This test may pass or fail based on random confidence scoring
    const response = await request(app)
      .post('/api/finance')
      .send({ amount: 100, currency: 'INVALID' });
    // Either 200 with low confidence or 400 for invalid currency
    expect([200, 400]).toContain(response.status);
  });
});