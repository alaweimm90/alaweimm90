import React from 'react';
import { TierType } from '@/components/ui/unified-pricing-card';
import { TIER_CONFIGS } from '@/constants/tiers';
import { MonthlyCoachingCard } from '@/components/ui/unified-pricing-card';
import { cn } from '@/lib/utils';

interface CentralizedTierCardProps {
  tier: TierType;
  billingCycle?: string;
  billingPeriod?: 'monthly' | 'quarterly' | 'annual';
  features?: string[];
  monthlyPrice?: number;
  yearlyPrice?: number;
  description?: string;
  isPopular?: boolean;
  onSelect: (tier: TierType) => void | Promise<void>;
  isCurrentTier?: boolean;
  className?: string;
}

export const CentralizedTierCard: React.FC<CentralizedTierCardProps> = ({
  tier,
  billingCycle,
  billingPeriod,
  features,
  monthlyPrice,
  yearlyPrice,
  description,
  isPopular,
  onSelect,
  isCurrentTier = false,
  className
}) => {
  // Find the tier config using the array structure or use passed props
  const tierConfig = TIER_CONFIGS.find(config => config.id === tier);
  
  // Use passed props if available, otherwise fall back to tier config
  const displayName = tierConfig?.displayName || `${tier} Program`;
  const tagline = description || tierConfig?.tagline || '';
  const featuresArray = features || tierConfig?.features || [];
  
  // Determine billing period from cycle or period prop
  const currentBillingPeriod = billingPeriod || (billingCycle === 'annual' ? 'annual' : 'monthly');
  
  // Calculate price - use passed props first, then tier config
  const getCurrentPrice = () => {
    if (monthlyPrice && yearlyPrice) {
      return currentBillingPeriod === 'annual' ? yearlyPrice : monthlyPrice;
    }
    
    if (tierConfig) {
      if (currentBillingPeriod === 'quarterly') {
        return tierConfig.quarterlyPrice;
      } else if (currentBillingPeriod === 'annual') {
        return tierConfig.annualPrice;
      }
      return tierConfig.monthlyPrice;
    }
    
    return 0;
  };

  const currentPrice = getCurrentPrice();
  const formatPrice = (amount: number) => amount.toFixed(0);

  return (
    <MonthlyCoachingCard
      title={displayName}
      subtitle={tagline}
      price={`$${formatPrice(currentPrice)}`}
      priceSubtext={currentBillingPeriod !== 'monthly' ? `Save money annually` : 'per month'}
      priceSubtextColor="text-green-400"
      tier={tier}
      badge={isPopular || tier === 'performance' ? 'popular' : tier === 'longevity' ? 'premium' : undefined}
      features={featuresArray.map(feature => ({
        text: feature,
        included: true,
        iconColor: tier === 'longevity' ? 'text-amber-400' : 'text-green-400'
      }))}
      buttonText={isCurrentTier ? "Current Plan" : "Choose Plan"}
      buttonVariant="primary"
      onClick={() => onSelect(tier)}
      isSelected={isCurrentTier}
      className={cn("w-full max-w-[360px]", className)}
    />
  );
};