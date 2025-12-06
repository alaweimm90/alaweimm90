import { 
  computeCompositeScore, 
  getScoringWeightsExplanation,
  hasCounterEvidence,
  computeOverallScore,
  overallSuspicion
} from '../src/lib/scoring';

describe('Scoring System', () => {
  describe('computeCompositeScore', () => {
    it('should compute normalized score with default weights', () => {
      const result = computeCompositeScore({
        gltrTail: 0.05,  // Low tail (suspicious)
        gltrVar: 0.08,   // Low variance (suspicious)
        curvature: -0.6, // Negative curvature (suspicious)
        watermarkP: 0.01, // Low p-value (watermark detected)
        refValidityRate: 0.3, // Low validity (suspicious)
        cwePerKloc: 5,   // High CWE density
        lengthChars: 1500,
        type: 'prose'
      });
      
      expect(result.score).toBeGreaterThan(0.5);
      expect(result.confidence).toBeDefined();
      expect(result.rationale.length).toBeGreaterThan(0);
    });

    it('should apply short segment penalty', () => {
      const longSegmentScore = computeCompositeScore({
        gltrTail: 0.1,
        curvature: -0.3,
        lengthChars: 1500
      });
      
      const shortSegmentScore = computeCompositeScore({
        gltrTail: 0.1,
        curvature: -0.3,
        lengthChars: 500  // Short segment
      });
      
      // Short segment should have penalty applied
      expect(shortSegmentScore.score).toBeGreaterThan(longSegmentScore.score);
      expect(shortSegmentScore.rationale).toContain('Short segment (<1000 chars)');
    });

    it('should handle weight normalization invariance', () => {
      const weights1 = {
        gltr: 0.22,
        detectgpt: 0.22,
        watermark: 0.18,
        citations: 0.25,
        cwe: 0.10,
        shortPenalty: 0.03
      };
      
      // Scale all weights by 10x
      const weights2 = {
        gltr: 2.2,
        detectgpt: 2.2,
        watermark: 1.8,
        citations: 2.5,
        cwe: 1.0,
        shortPenalty: 0.3
      };
      
      const score1 = computeCompositeScore({
        gltrTail: 0.1,
        curvature: -0.3,
        weights: weights1
      });
      
      const score2 = computeCompositeScore({
        gltrTail: 0.1,
        curvature: -0.3,
        weights: weights2
      });
      
      // Scores should be the same after normalization
      expect(Math.abs(score1.score - score2.score)).toBeLessThan(0.01);
    });

    it('should ignore citation signals for code segments', () => {
      const result = computeCompositeScore({
        refValidityRate: 0.0, // Bad citations
        type: 'code',
        lengthChars: 1500
      });
      
      // Citation signal should not affect code segments
      expect(result.rationale).not.toContain('citation');
    });

    it('should ignore CWE signals for non-code segments', () => {
      const result = computeCompositeScore({
        cwePerKloc: 10, // High CWE density
        type: 'prose',
        lengthChars: 1500
      });
      
      // CWE signal should not affect prose segments
      expect(result.rationale).not.toContain('CWE');
    });
  });

  describe('getScoringWeightsExplanation', () => {
    it('should format default weights correctly', () => {
      const explanation = getScoringWeightsExplanation();
      
      expect(explanation).toContain('GLTR (22%)');
      expect(explanation).toContain('DetectGPT (22%)');
      expect(explanation).toContain('Citations (25%)');
      expect(explanation).toContain('Watermark (18%)');
      expect(explanation).toContain('Code Security (10%)');
      expect(explanation).toContain('Length penalty (3%)');
    });

    it('should format custom weights correctly', () => {
      const weights = {
        gltr: 1,
        detectgpt: 1,
        watermark: 1,
        citations: 1,
        cwe: 1,
        shortPenalty: 1
      };
      
      const explanation = getScoringWeightsExplanation(weights);
      
      // All weights equal, should be ~17% each
      expect(explanation).toMatch(/GLTR \(1[67]%\)/);
      expect(explanation).toMatch(/DetectGPT \(1[67]%\)/);
    });
  });

  describe('hasCounterEvidence', () => {
    it('should detect positive curvature as counter-evidence', () => {
      expect(hasCounterEvidence(0.1)).toBe(true);
      expect(hasCounterEvidence(0)).toBe(true);
      expect(hasCounterEvidence(-0.1)).toBe(false);
      expect(hasCounterEvidence(undefined)).toBe(false);
    });
  });

  describe('computeOverallScore', () => {
    it('should compute length-weighted mean correctly', () => {
      const scores = {
        'seg1': { score: 0.8 },
        'seg2': { score: 0.6 },
        'seg3': { score: 0.4 }
      };
      
      const segments = [
        { segmentId: 'seg1', lengthChars: 2000 }, // Weight: 2000
        { segmentId: 'seg2', lengthChars: 1500 }, // Weight: 1500
        { segmentId: 'seg3', lengthChars: 500 }   // Weight: 250 (half due to <1000)
      ];
      
      const overall = computeOverallScore(scores, segments);
      
      // Expected: (0.8*2000 + 0.6*1500 + 0.4*250) / (2000+1500+250)
      // = (1600 + 900 + 100) / 3750 = 2600 / 3750 = 0.693...
      expect(overall).toBeCloseTo(0.693, 2);
    });

    it('should apply 0.5 weight factor for short segments', () => {
      const scores = {
        'short': { score: 1.0 },
        'long': { score: 0.5 }
      };
      
      const segments = [
        { segmentId: 'short', lengthChars: 500 },  // Weight: 500 * 0.5 = 250
        { segmentId: 'long', lengthChars: 2000 }   // Weight: 2000 * 1 = 2000
      ];
      
      const overall = computeOverallScore(scores, segments);
      
      // (1.0*250 + 0.5*2000) / (250+2000) = 1250 / 2250 = 0.556
      expect(overall).toBeCloseTo(0.556, 2);
    });

    it('should handle missing scores gracefully', () => {
      const scores = {
        'seg1': { score: 0.7 }
        // seg2 is missing
      };
      
      const segments = [
        { segmentId: 'seg1', lengthChars: 1000 },
        { segmentId: 'seg2', lengthChars: 1000 }
      ];
      
      const overall = computeOverallScore(scores, segments);
      
      // Should only use seg1
      expect(overall).toBe(0.7);
    });
  });

  describe('overallSuspicion', () => {
    it('should match computeOverallScore implementation', () => {
      const report = {
        segments: [
          { segmentId: 'seg1', lengthChars: 2000 },
          { segmentId: 'seg2', lengthChars: 800 },
          { segmentId: 'seg3', lengthChars: 1200 }
        ],
        scores: {
          'seg1': { score: 0.9 },
          'seg2': { score: 0.5 },
          'seg3': { score: 0.7 }
        }
      };
      
      const suspicion = overallSuspicion(report);
      const overall = computeOverallScore(report.scores, report.segments);
      
      expect(suspicion).toBeCloseTo(overall, 10);
    });

    it('should handle empty report', () => {
      const report = {
        segments: [],
        scores: {}
      };
      
      const suspicion = overallSuspicion(report);
      expect(suspicion).toBe(0);
    });
  });
});