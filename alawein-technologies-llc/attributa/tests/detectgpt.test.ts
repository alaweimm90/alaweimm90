import { detectGPTCurvature } from '../src/lib/nlp/detectgpt';

describe('DetectGPT Analysis', () => {
  // Mock log probability function
  const mockGetLogProb = async (text: string): Promise<number> => {
    // Simulate different log probabilities based on text characteristics
    const hasAIMarkers = text.includes('Furthermore') || 
                        text.includes('Moreover') || 
                        text.includes('In conclusion') ||
                        text.includes('machine learning') ||
                        text.includes('algorithms');
    
    const isPerturbation = text.includes('[MASK]') || text.length < 100;
    
    // For AI text: original should have higher log prob, perturbations should have even higher
    // This creates negative curvature (original - perturbations < 0)
    if (hasAIMarkers && !isPerturbation) {
      return -2.0; // Lower log prob for original AI text
    }
    
    if (hasAIMarkers && isPerturbation) {
      return -1.5; // Higher log prob for perturbed AI text (this creates negative curvature)
    }
    
    // Human text: original and perturbations should be closer
    const baseLogProb = isPerturbation ? -2.3 : -2.5;
    const noise = (Math.random() - 0.5) * 0.2;
    return baseLogProb + noise;
  };

  describe('detectGPTCurvature', () => {
    it('should return negative curvature for LM-generated text', async () => {
      const aiGeneratedText = `Furthermore, it is important to note that machine learning models have revolutionized various industries across the globe. Moreover, these technological advancements have led to unprecedented improvements in operational efficiency and cost reduction. Additionally, the implementation of artificial intelligence systems has enabled organizations to automate complex processes that were previously considered impossible to streamline. In conclusion, the transformative impact of artificial intelligence cannot be overstated in today's digital landscape.`;
      
      const result = await detectGPTCurvature(
        aiGeneratedText,
        mockGetLogProb,
        20 // Fewer perturbations for testing
      );
      
      expect(result.curvature).toBeLessThan(0);
      expect(result.numPerturbations).toBeGreaterThan(0);
    });

    it('should return near-zero or positive curvature for human-edited text', async () => {
      const humanText = `So I was thinking about this problem yesterday and it really got me confused for a while. It's kinda weird how things work sometimes, you know? Like, the solution wasn't obvious at first but then it just clicked and everything made sense. I tried a bunch of different approaches before settling on this one. Anyway, that's my take on it - sometimes you just gotta keep trying until something works. The whole experience taught me a lot about persistence and creative problem solving.`;
      
      const result = await detectGPTCurvature(
        humanText,
        mockGetLogProb,
        20
      );
      
      // Human text should have curvature closer to zero or positive
      expect(result.curvature).toBeGreaterThan(-0.3);
      expect(result.numPerturbations).toBeGreaterThan(0);
    });

    it('should return zero curvature for text shorter than 300 chars', async () => {
      const shortText = 'This is a short text with only a few words.';
      
      const result = await detectGPTCurvature(
        shortText,
        mockGetLogProb,
        50
      );
      
      expect(result.curvature).toBe(0);
      expect(result.numPerturbations).toBe(0);
    });

    it('should respect max perturbation limit', async () => {
      const text = 'A'.repeat(500); // Long enough text
      
      const result = await detectGPTCurvature(
        text,
        mockGetLogProb,
        150 // Request more than max (100)
      );
      
      expect(result.numPerturbations).toBeLessThanOrEqual(20); // Limited in implementation
    });

    it('should handle errors gracefully', async () => {
      const errorLogProb = async () => {
        throw new Error('API error');
      };
      
      const text = 'This text will cause an error in log prob calculation.'.repeat(10);
      
      const result = await detectGPTCurvature(
        text,
        errorLogProb,
        10
      );
      
      expect(result.curvature).toBe(0);
      expect(result.numPerturbations).toBe(0);
    });

    it('should show different curvature for different perturbation counts', async () => {
      const text = `Machine learning has fundamentally transformed how we approach complex computational problems in the modern era. The sophisticated algorithms can identify intricate patterns and relationships that humans might miss due to cognitive limitations. This remarkable capability has extensive applications across numerous fields including healthcare, finance, transportation, and scientific research. Furthermore, the continuous advancement of these technologies promises even greater innovations in the future.`;
      
      const result1 = await detectGPTCurvature(text, mockGetLogProb, 10);
      const result2 = await detectGPTCurvature(text, mockGetLogProb, 20);
      
      // Both should detect AI-like characteristics
      expect(result1.curvature).toBeLessThan(0);
      expect(result2.curvature).toBeLessThan(0);
      
      // Different perturbation counts
      expect(result1.numPerturbations).toBeLessThan(result2.numPerturbations);
    });
  });
});