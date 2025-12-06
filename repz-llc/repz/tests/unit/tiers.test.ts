/**
 * Unit tests for tier system configuration
 * @file tests/unit/tiers.test.ts
 */

import { describe, it, expect, beforeEach } from 'vitest';

// Mock the tier constants (these would be imported from the actual file)
const TIER_ORDER = ['core', 'adaptive', 'performance', 'longevity'] as const;
type TierType = typeof TIER_ORDER[number];

interface TierConfig {
  id: TierType;
  name: string;
  monthlyPrice: number;
  annualPrice: number;
  features: string[];
  color: string;
}

const TIER_CONFIGS: Record<TierType, TierConfig> = {
  core: {
    id: 'core',
    name: 'Core Program',
    monthlyPrice: 89,
    annualPrice: 71.20,
    features: ['Training program', 'Nutrition plan', '72h response time'],
    color: '#3B82F6',
  },
  adaptive: {
    id: 'adaptive',
    name: 'Adaptive Engine',
    monthlyPrice: 149,
    annualPrice: 119.20,
    features: ['Biomarkers', 'Wearable integration', 'Weekly check-ins', '48h response'],
    color: '#8B5CF6',
  },
  performance: {
    id: 'performance',
    name: 'Performance Suite',
    monthlyPrice: 229,
    annualPrice: 183.20,
    features: ['AI assistant', 'Form analysis', 'PEDs protocols', '24h response'],
    color: '#F97316',
  },
  longevity: {
    id: 'longevity',
    name: 'Longevity Concierge',
    monthlyPrice: 349,
    annualPrice: 279.20,
    features: ['In-person training', 'Concierge service', '12h response'],
    color: '#10B981',
  },
};

// Utility functions to test
function getTierIndex(tier: TierType): number {
  return TIER_ORDER.indexOf(tier);
}

function hasMinimumTier(userTier: TierType, requiredTier: TierType): boolean {
  return getTierIndex(userTier) >= getTierIndex(requiredTier);
}

function calculateAnnualSavings(tier: TierType): number {
  const config = TIER_CONFIGS[tier];
  const monthlyTotal = config.monthlyPrice * 12;
  const annualTotal = config.annualPrice * 12;
  return monthlyTotal - annualTotal;
}

function getAnnualDiscountPercentage(tier: TierType): number {
  const config = TIER_CONFIGS[tier];
  return Math.round((1 - config.annualPrice / config.monthlyPrice) * 100);
}

function isValidTier(tier: string): tier is TierType {
  return TIER_ORDER.includes(tier as TierType);
}

// Tests
describe('Tier System', () => {
  describe('TIER_ORDER', () => {
    it('should have exactly 4 tiers', () => {
      expect(TIER_ORDER).toHaveLength(4);
    });

    it('should be in correct order from lowest to highest', () => {
      expect(TIER_ORDER[0]).toBe('core');
      expect(TIER_ORDER[1]).toBe('adaptive');
      expect(TIER_ORDER[2]).toBe('performance');
      expect(TIER_ORDER[3]).toBe('longevity');
    });
  });

  describe('TIER_CONFIGS', () => {
    it('should have config for each tier', () => {
      TIER_ORDER.forEach((tier) => {
        expect(TIER_CONFIGS[tier]).toBeDefined();
      });
    });

    it('should have correct pricing for core tier', () => {
      expect(TIER_CONFIGS.core.monthlyPrice).toBe(89);
      expect(TIER_CONFIGS.core.annualPrice).toBe(71.20);
    });

    it('should have correct pricing for adaptive tier', () => {
      expect(TIER_CONFIGS.adaptive.monthlyPrice).toBe(149);
      expect(TIER_CONFIGS.adaptive.annualPrice).toBe(119.20);
    });

    it('should have correct pricing for performance tier', () => {
      expect(TIER_CONFIGS.performance.monthlyPrice).toBe(229);
      expect(TIER_CONFIGS.performance.annualPrice).toBe(183.20);
    });

    it('should have correct pricing for longevity tier', () => {
      expect(TIER_CONFIGS.longevity.monthlyPrice).toBe(349);
      expect(TIER_CONFIGS.longevity.annualPrice).toBe(279.20);
    });

    it('should have features array for each tier', () => {
      TIER_ORDER.forEach((tier) => {
        expect(Array.isArray(TIER_CONFIGS[tier].features)).toBe(true);
        expect(TIER_CONFIGS[tier].features.length).toBeGreaterThan(0);
      });
    });

    it('should have valid hex color for each tier', () => {
      const hexColorRegex = /^#[0-9A-Fa-f]{6}$/;
      TIER_ORDER.forEach((tier) => {
        expect(TIER_CONFIGS[tier].color).toMatch(hexColorRegex);
      });
    });
  });

  describe('getTierIndex', () => {
    it('should return 0 for core', () => {
      expect(getTierIndex('core')).toBe(0);
    });

    it('should return 1 for adaptive', () => {
      expect(getTierIndex('adaptive')).toBe(1);
    });

    it('should return 2 for performance', () => {
      expect(getTierIndex('performance')).toBe(2);
    });

    it('should return 3 for longevity', () => {
      expect(getTierIndex('longevity')).toBe(3);
    });
  });

  describe('hasMinimumTier', () => {
    it('should return true when user tier equals required tier', () => {
      expect(hasMinimumTier('core', 'core')).toBe(true);
      expect(hasMinimumTier('adaptive', 'adaptive')).toBe(true);
      expect(hasMinimumTier('performance', 'performance')).toBe(true);
      expect(hasMinimumTier('longevity', 'longevity')).toBe(true);
    });

    it('should return true when user tier is higher than required', () => {
      expect(hasMinimumTier('adaptive', 'core')).toBe(true);
      expect(hasMinimumTier('performance', 'core')).toBe(true);
      expect(hasMinimumTier('performance', 'adaptive')).toBe(true);
      expect(hasMinimumTier('longevity', 'core')).toBe(true);
      expect(hasMinimumTier('longevity', 'adaptive')).toBe(true);
      expect(hasMinimumTier('longevity', 'performance')).toBe(true);
    });

    it('should return false when user tier is lower than required', () => {
      expect(hasMinimumTier('core', 'adaptive')).toBe(false);
      expect(hasMinimumTier('core', 'performance')).toBe(false);
      expect(hasMinimumTier('core', 'longevity')).toBe(false);
      expect(hasMinimumTier('adaptive', 'performance')).toBe(false);
      expect(hasMinimumTier('adaptive', 'longevity')).toBe(false);
      expect(hasMinimumTier('performance', 'longevity')).toBe(false);
    });
  });

  describe('calculateAnnualSavings', () => {
    it('should calculate correct savings for core tier', () => {
      const savings = calculateAnnualSavings('core');
      expect(savings).toBeCloseTo(213.60, 2); // (89 - 71.20) * 12
    });

    it('should calculate correct savings for adaptive tier', () => {
      const savings = calculateAnnualSavings('adaptive');
      expect(savings).toBeCloseTo(357.60, 2); // (149 - 119.20) * 12
    });

    it('should calculate correct savings for performance tier', () => {
      const savings = calculateAnnualSavings('performance');
      expect(savings).toBeCloseTo(549.60, 2); // (229 - 183.20) * 12
    });

    it('should calculate correct savings for longevity tier', () => {
      const savings = calculateAnnualSavings('longevity');
      expect(savings).toBeCloseTo(837.60, 2); // (349 - 279.20) * 12
    });
  });

  describe('getAnnualDiscountPercentage', () => {
    it('should return 20% discount for all tiers', () => {
      TIER_ORDER.forEach((tier) => {
        expect(getAnnualDiscountPercentage(tier)).toBe(20);
      });
    });
  });

  describe('isValidTier', () => {
    it('should return true for valid tiers', () => {
      expect(isValidTier('core')).toBe(true);
      expect(isValidTier('adaptive')).toBe(true);
      expect(isValidTier('performance')).toBe(true);
      expect(isValidTier('longevity')).toBe(true);
    });

    it('should return false for invalid tiers', () => {
      expect(isValidTier('free')).toBe(false);
      expect(isValidTier('premium')).toBe(false);
      expect(isValidTier('')).toBe(false);
      expect(isValidTier('CORE')).toBe(false); // case sensitive
    });
  });
});

describe('Tier Pricing Consistency', () => {
  it('should have increasing prices from core to longevity', () => {
    const prices = TIER_ORDER.map((tier) => TIER_CONFIGS[tier].monthlyPrice);
    for (let i = 1; i < prices.length; i++) {
      expect(prices[i]).toBeGreaterThan(prices[i - 1]);
    }
  });

  it('should have annual price less than monthly price for all tiers', () => {
    TIER_ORDER.forEach((tier) => {
      expect(TIER_CONFIGS[tier].annualPrice).toBeLessThan(TIER_CONFIGS[tier].monthlyPrice);
    });
  });

  it('should have consistent 20% annual discount across all tiers', () => {
    TIER_ORDER.forEach((tier) => {
      const config = TIER_CONFIGS[tier];
      const expectedAnnual = config.monthlyPrice * 0.8;
      expect(config.annualPrice).toBeCloseTo(expectedAnnual, 2);
    });
  });
});
