import { TierType } from '@/types/fitness';

// Removed legacy tier types - only use current tier structure

/**
 * Current tier mapping for validation
 */
export const TIER_MAPPING: Record<string, TierType> = {
  'core': 'core',
  'adaptive': 'adaptive',
  'performance': 'performance',
  'longevity': 'longevity'
};

/**
 * Validates and normalizes tier names to current structure
 */
export function validateTier(tier: string | TierType): TierType {
  if (!tier) return 'core';
  
  const mappedTier = TIER_MAPPING[tier.toLowerCase()];
  return mappedTier || 'core';
}

/**
 * Check if a tier is valid in the new structure
 */
export function isValidNewTier(tier: string): tier is TierType {
  return ['core', 'adaptive', 'performance', 'longevity'].includes(tier);
}

/**
 * Safely convert any tier reference to the current structure
 */
export function normalizeTier(tier: unknown): TierType {
  if (typeof tier !== 'string') return 'core';
  return validateTier(tier);
}

/**
 * Get the display name for a tier
 */
export function getTierDisplayName(tier: TierType): string {
  const displayNames = {
    core: 'Core Program',
    adaptive: 'Adaptive Engine',
    performance: 'Prime Suite',
    longevity: 'Elite Concierge'
  };
  return displayNames[tier];
}