// Centralized Tier Card Component - Uses your existing UnifiedTierCard
// This is just a re-export to maintain centralized import paths

import React from 'react';
import { UnifiedTierCard } from '@/components/ui/unified-tier-card';

export { 
  UnifiedTierCard as TierCard,
  UnifiedTierCard as CentralizedTierCard
} from '@/components/ui/unified-tier-card';

// Export the props type from the original component
export type TierCardProps = React.ComponentProps<typeof UnifiedTierCard>;
export type CentralizedTierCardProps = TierCardProps;