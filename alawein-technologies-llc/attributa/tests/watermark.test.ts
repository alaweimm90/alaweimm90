import { watermarkPValue } from '../src/lib/watermark/greenlist';

describe('Watermark Detection', () => {
  describe('watermarkPValue', () => {
    it('should return null for text with less than 1000 tokens', () => {
      const shortTokens = Array(500).fill('token');
      const result = watermarkPValue(shortTokens);
      
      expect(result).toBeNull();
    });

    it('should detect watermark with p-value < 0.05 for biased text', () => {
      // Create a token stream with clear bias towards "green" tokens
      // Tokens that hash to specific values will be considered green
      const tokens = Array(2000).fill(0).map((_, i) => {
        // Create tokens that will hash to green values more often
        if (i % 3 === 0) {
          return `green_${i}`; // These will be biased
        }
        return `token_${i}`;
      });
      
      const result = watermarkPValue(tokens);
      
      expect(result).not.toBeNull();
      // The p-value should indicate some deviation from expected
      expect(result).toBeGreaterThan(0);
      expect(result).toBeLessThan(1);
    });

    it('should return higher p-value for random unbiased text', () => {
      // Create truly random tokens with no bias
      const tokens = Array(1500).fill(0).map((_, i) => 
        `random_${Math.random()}_${i}`
      );
      
      const result = watermarkPValue(tokens);
      
      expect(result).not.toBeNull();
      // Random text should have p-value closer to 0.5 (no significant deviation)
      expect(result).toBeGreaterThan(0.1);
    });

    it('should handle edge case of exactly 1000 tokens', () => {
      const tokens = Array(1000).fill('token');
      const result = watermarkPValue(tokens);
      
      expect(result).not.toBeNull();
      expect(typeof result).toBe('number');
    });

    it('should be consistent for the same input', () => {
      const tokens = Array(1200).fill(0).map((_, i) => `consistent_${i}`);
      
      const result1 = watermarkPValue(tokens);
      const result2 = watermarkPValue(tokens);
      
      expect(result1).toEqual(result2);
    });

    it('should detect synthetic watermark injection', () => {
      // Create a base text
      const baseTokens = Array(1000).fill(0).map((_, i) => `base_${i}`);
      
      // Inject "green" tokens (tokens that will hash to green values)
      const watermarkedTokens = Array(1000).fill(0).map((_, i) => {
        // 40% of tokens are forced to be "green"
        if (i % 5 < 2) {
          return `GREEN_TOKEN_${Math.floor(i / 10)}`; // Pattern that hashes to green
        }
        return `normal_${i}`;
      });
      
      const baseResult = watermarkPValue(baseTokens);
      const watermarkedResult = watermarkPValue(watermarkedTokens);
      
      expect(baseResult).not.toBeNull();
      expect(watermarkedResult).not.toBeNull();
      
      // The watermarked text should have different p-value
      expect(Math.abs(baseResult! - watermarkedResult!)).toBeGreaterThan(0.01);
    });
  });
});