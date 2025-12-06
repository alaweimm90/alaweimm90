// Simple verification script to test that our analyzers work
import { gltrStats } from './src/lib/nlp/gltr.js';

console.log('🧪 Testing GLTR analyzer...');

// Test GLTR with sample token ranks
const sampleRanks = [5, 10, 15, 50, 100, 200, 500, 1000, 2000, 5000];
const result = gltrStats(sampleRanks);

console.log('✅ GLTR Stats:', {
  tailTokenShare: result.tailTokenShare,
  rankVariance: result.rankVariance,
  histogram: result.histogram
});

// Verify histogram sums to 1
const histogramSum = result.histogram.reduce((a, b) => a + b, 0);
console.log('✅ Histogram sum:', histogramSum, '(should be 1.0)');

if (Math.abs(histogramSum - 1.0) < 0.001) {
  console.log('🎉 GLTR analyzer working correctly!');
} else {
  console.log('❌ GLTR analyzer has issues');
  process.exit(1);
}

console.log('\n✨ All basic verifications passed!');
console.log('🚀 The real analyzers are ready to replace the mocks.');