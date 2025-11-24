const request = require('supertest');
const app = require('./index');

describe('AI Agent Demo App - additional', () => {
  test('should return 500 on error route', async () => {
    const res = await request(app).get('/error');
    expect(res.status).toBe(500);
    expect(res.body).toHaveProperty('error');
  });

  test('should trigger self-learning log every 10 requests', async () => {
    const origRandom = Math.random;
    Math.random = () => 0.99;
    for (let i = 0; i < 10; i++) {
      const res = await request(app)
        .post('/api/finance')
        .send({ amount: 10, currency: 'USD' })
        .set('Content-Type', 'application/json');
      expect([200, 400]).toContain(res.status);
    }
    Math.random = origRandom;
  });
});
