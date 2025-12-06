// tests/gltr.test.ts
import { gltrStats } from '@/lib/nlp/gltr';

describe('gltrStats', () => {
  it('histogram sums to ~1', () => {
    const ranks = [5, 50, 500, 5000, 12, 120, 1200, 9000];
    const { histogram } = gltrStats(ranks);
    const sum = histogram.reduce((a, b) => a + b, 0);
    expect(sum).toBeGreaterThan(0.99);
    expect(sum).toBeLessThan(1.01);
  });
});
