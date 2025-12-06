import React from 'react';
import { Card } from '@/ui/molecules/Card';

interface TierCardProps {
  tier?: string;
  className?: string;
  children?: React.ReactNode;
  billingCycle?: string;
  billingPeriod?: string;
  variant?: string;
  onSelect?: () => void;
  onSelectTier?: (tierId: string) => void;
  showFeatures?: boolean;
  currentTier?: string;
}

export const TierCard: React.FC<TierCardProps> = ({ 
  tier, 
  className, 
  children,
  billingCycle,
  billingPeriod,
  variant,
  onSelect,
  onSelectTier,
  showFeatures,
  currentTier
}) => {
  const handleClick = () => {
    onSelect?.();
    onSelectTier?.(tier);
  };

  return (
    <Card className={className} onClick={handleClick}>
      <div className="p-4">
        {tier && <h3 className="font-semibold mb-2">{tier}</h3>}
        {children}
        {showFeatures && (
          <div className="mt-2">
            <p className="text-sm text-muted-foreground">
              {variant} tier with {billingCycle || billingPeriod} billing
            </p>
          </div>
        )}
      </div>
    </Card>
  );
};

// Export alternative names for compatibility
export const UnifiedTierCard = TierCard;
export const EnhancedTierDisplay = TierCard;
export const EnhancedTierCard = TierCard;

export default TierCard;