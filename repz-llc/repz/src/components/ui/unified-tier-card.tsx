import React from 'react';
import { Check, Star, Zap, Crown, Award } from 'lucide-react';
import { TierType, getTierConfigByType, BillingCycle } from '@/constants/tiers';
import { getStripePrice, BILLING_CYCLE_CONFIG } from '@/constants/stripeConfig';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { FloatingParticles } from '@/components/ui/floating-particles';
import { ComingSoonOverlay } from '@/components/ui/coming-soon-overlay';
import { cn } from '@/lib/utils';

interface UnifiedTierCardProps {
  tier: TierType;
  billingCycle?: BillingCycle;
  variant?: 'standard' | 'compact' | 'detailed' | 'comparison' | 'conversion' | 'glass' | 'floating';
  showFeatures?: boolean;
  showBadge?: boolean;
  showSavings?: boolean;
  showConversionPsychology?: boolean;
  interactive?: boolean;
  className?: string;
  onSelect?: (tier: TierType) => void;
  // Legacy compatibility for ConversionTierCard
  onSelectTier?: (tier: TierType, billing: BillingCycle) => void;
  isCurrentTier?: boolean;
  // Enhanced props for conversion optimization
  showPopularBadge?: boolean;
  showScarcity?: boolean;
  showSocialProof?: boolean;
  // Coming soon functionality
  comingSoon?: boolean;
  comingSoonMessage?: string;
}

// This ONE component replaces all your tier card variants
export const UnifiedTierCard: React.FC<UnifiedTierCardProps> = ({
  tier,
  billingCycle = 'monthly',
  variant = 'standard',
  showFeatures = true,
  showBadge = true,
  interactive = true,
  className = '',
  onSelect,
  onSelectTier,
  isCurrentTier = false,
  comingSoon = false,
  comingSoonMessage
}) => {
  const config = getTierConfigByType(tier);
  
  // Handle both callback patterns for backwards compatibility
  const handleSelect = () => {
    if (onSelectTier) {
      onSelectTier(tier, billingCycle);
    } else if (onSelect) {
      onSelect(tier);
    }
  };
  
  // Get price based on billing cycle
  const getPrice = () => {
    switch (billingCycle) {
      case 'quarterly': return config.quarterlyPrice;
      case 'semiannual': return config.semiannualPrice;
      case 'annual': return config.annualPrice;
      default: return config.monthlyPrice;
    }
  };

  // Get savings percentage
  const getSavings = () => {
    if (billingCycle === 'monthly') return 0;
    return config.savings[billingCycle as keyof typeof config.savings] || 0;
  };

  const price = getPrice();
  const savings = getSavings();
  
  // Badge icon based on tier
  const getBadgeIcon = () => {
    switch (tier) {
      case 'adaptive': return <Zap className="w-4 h-4" />;
      case 'performance': return <Star className="w-4 h-4" />;
      case 'longevity': return <Crown className="w-4 h-4" />;
      default: return null;
    }
  };

  // Glass morphism classes with tier-specific theming
  const getGlassClasses = () => {
    const tierGlassClass = `glass-tier-card-${tier}`;
    return `glass-tier-card ${tierGlassClass} ${config.isPopular ? 'ring-2 ring-opacity-50' : ''}`;
  };

  // Base classes for all variants with premium glass morphism
  const baseClasses = `
    ${getGlassClasses()}
    relative rounded-2xl transition-all duration-300 animate-elegant-float
    ${interactive ? 'cursor-pointer' : ''}
  `;

  // Variant-specific rendering
  const renderContent = () => {
    switch (variant) {
      case 'compact':
        return (
          <div className={`${baseClasses} p-4 ${className}`} onClick={handleSelect}>
            <div className="flex justify-between items-center">
              <div>
                <h3 className="font-semibold text-lg">{config.displayName}</h3>
                <p className="text-2xl font-bold">${price}<span className="text-sm text-gray-500">/mo</span></p>
              </div>
              {showBadge && config.badge && (
                <span className={`px-2 py-1 text-xs font-medium rounded-full bg-gradient-to-r ${config.gradient} text-white`}>
                  {config.badge}
                </span>
              )}
            </div>
          </div>
        );

      case 'detailed':
        return (
          <div className={`${baseClasses} p-8 ${className}`} onClick={handleSelect}>
            <FloatingParticles tier={tier} count={3} />
            <div className="text-center mb-6">
              {showBadge && config.badge && (
                <div className="flex justify-center mb-4">
                  <span className={`inline-flex items-center gap-1 px-3 py-1 text-sm font-medium rounded-full bg-gradient-to-r ${config.gradient} text-white`}>
                    {getBadgeIcon()}
                    {config.badge}
                  </span>
                </div>
              )}
              
              <h3 className="text-2xl font-bold mb-2">{config.displayName}</h3>
              <p className="text-gray-600 mb-4">{config.tagline}</p>
              <p className="text-sm text-gray-500 mb-6">{config.description}</p>
              
              <div className="mb-6">
                <div className="text-4xl font-bold mb-2">
                  ${price}
                  <span className="text-lg text-gray-500 font-normal">/month</span>
                </div>
                {savings > 0 && (
                  <p className="text-green-600 font-medium">
                    Save {savings}% with {billingCycle} billing
                  </p>
                )}
              </div>
            </div>

            {showFeatures && (
              <div className="mb-8">
                <ul className="space-y-3">
                  {config.features.map((feature, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                      <span className="text-sm">{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <button 
              className={`w-full py-3 px-6 rounded-lg font-semibold transition-all bg-gradient-to-r ${config.gradient} text-white hover:opacity-90 ${isCurrentTier ? 'opacity-50 cursor-not-allowed' : ''}`}
              onClick={(e) => {
                e.stopPropagation();
                if (!isCurrentTier) handleSelect();
              }}
              disabled={isCurrentTier}
            >
              {isCurrentTier ? 'Current Plan' : config.ctaText}
            </button>
          </div>
        );

      case 'comparison':
        return (
          <div className={`${baseClasses} p-6 ${className}`}>
            <div className="text-center mb-4">
              {showBadge && config.badge && (
                <span className={`inline-block px-2 py-1 text-xs font-medium rounded-full bg-gradient-to-r ${config.gradient} text-white mb-2`}>
                  {config.badge}
                </span>
              )}
              <h3 className="text-xl font-bold">{config.displayName}</h3>
              <div className="text-2xl font-bold mt-2">
                ${price}<span className="text-sm text-gray-500">/mo</span>
              </div>
            </div>

            {showFeatures && (
              <ul className="space-y-2">
                {config.features.slice(0, 5).map((feature, index) => (
                  <li key={index} className="flex items-center gap-2">
                    <Check className="w-4 h-4 text-green-500" />
                    <span className="text-sm">{feature}</span>
                  </li>
                ))}
                {config.features.length > 5 && (
                  <li className="text-sm text-gray-500">
                    +{config.features.length - 5} more features
                  </li>
                )}
              </ul>
            )}
          </div>
        );

      default: // 'standard'
        return (
          <Card className={cn(baseClasses, className)} onClick={handleSelect}>
            <CardHeader className="text-center">
              {showBadge && config.badge && (
                <div className="flex justify-center mb-3">
                  <Badge className={`bg-gradient-to-r ${config.gradient} text-white border-0`}>
                    {getBadgeIcon()}
                    <span className="ml-1">{config.badge}</span>
                  </Badge>
                </div>
              )}
              
              <CardTitle className="text-xl mb-2">{config.displayName}</CardTitle>
              <p className="text-gray-600 text-sm mb-4">{config.tagline}</p>
              
              <div className="mb-4">
                <div className="text-3xl font-bold">
                  ${price}
                  <span className="text-base text-gray-500 font-normal">/mo</span>
                </div>
                {savings > 0 && (
                  <p className="text-sm text-green-600 font-medium">
                    {savings}% savings
                  </p>
                )}
              </div>
            </CardHeader>

            <CardContent>
              {showFeatures && (
                <div className="mb-6">
                  <ul className="space-y-2">
                    {config.features.slice(0, 6).map((feature, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <Check className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
                        <span className="text-sm">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <Button 
                className={`w-full bg-gradient-to-r ${config.gradient} text-white hover:opacity-90 ${isCurrentTier ? 'opacity-50 cursor-not-allowed' : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  if (!isCurrentTier) handleSelect();
                }}
                disabled={isCurrentTier}
              >
                {isCurrentTier ? 'Current Plan' : config.ctaText}
              </Button>
            </CardContent>
          </Card>
        );
    }
  };

  return (
    <ComingSoonOverlay 
      disabled={comingSoon} 
      message={comingSoonMessage}
      className="w-full h-full"
    >
      {renderContent()}
    </ComingSoonOverlay>
  );
};

// Export additional variants for backwards compatibility
export const TierCard = UnifiedTierCard;
export const ConsolidatedTierCard = UnifiedTierCard;
export const EnhancedTierCard = UnifiedTierCard;
export const InteractiveTierCard = UnifiedTierCard;
export const PricingPlanCard = UnifiedTierCard;
export const ConversionTierCard = UnifiedTierCard;

export default UnifiedTierCard;