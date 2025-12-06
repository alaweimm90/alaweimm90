import React from 'react';
import { Check, X, Crown, Star, Zap, Sprout, TrendingUp } from 'lucide-react';
import { Button } from './unified-button';
import { Badge } from '@/ui/atoms/Badge';
import { cn } from '@/lib/utils';
import type { BillingPeriod } from './pricing-toggle';

/* =====================================================================
   PREMIUM COMPARISON TABLE - ENTERPRISE GRADE
   Glass morphism + mathematical color system + premium animations
   ===================================================================== */

interface TierData {
  id: string;
  name: string;
  description: string;
  price: {
    monthly: number;
    quarterly: number;
    semiannual: number;
    annual: number;
  };
  savings: {
    quarterly: number;
    semiannual: number; 
    annual: number;
  };
  popular?: boolean;
  premium?: boolean;
  features: TierFeature[];
  icon: React.ReactNode;
}

interface TierFeature {
  name: string;
  included: boolean;
  highlight?: boolean;
}

interface FeatureCategory {
  title: string;
  features: {
    name: string;
    values: string[];
  }[];
}

interface PremiumComparisonTableProps {
  tierData: TierData[];
  featureCategories: FeatureCategory[];
  billingPeriod: BillingPeriod;
  onSelectTier?: (tierId: string) => void;
  currentTier?: string;
  className?: string;
}

const getTierIcon = (tierId: string) => {
  switch (tierId) {
    case 'core': return <Sprout className="h-5 w-5" />;
    case 'adaptive': return <TrendingUp className="h-5 w-5" />;
    case 'performance': return <Zap className="h-5 w-5" />;
    case 'longevity': return <Crown className="h-5 w-5" />;
    default: return <Sprout className="h-5 w-5" />;
  }
};

// ENTERPRISE: ALL CTA buttons use primary orange for consistent action hierarchy
const getTierButtonVariant = (tierId: string) => {
  return 'primary'; // All tier buttons use primary (orange) variant
};

const getPriceForPeriod = (tier: TierData, period: BillingPeriod) => {
  const priceMap = { 
    monthly: tier.price.monthly, 
    quarterly: tier.price.quarterly, 
    'semi-annual': tier.price.semiannual, 
    annual: tier.price.annual 
  };
  return priceMap[period];
};

const getSavingsForPeriod = (tier: TierData, period: BillingPeriod) => {
  if (period === 'monthly') return 0;
  const savingsMap = { 
    quarterly: tier.savings.quarterly, 
    'semi-annual': tier.savings.semiannual, 
    annual: tier.savings.annual 
  };
  return savingsMap[period] || 0;
};

const renderFeatureValue = (value: string, tierIndex: number, tierId: string) => {
  if (value === "✓") {
    // All checkmarks green except longevity (yellow to match cards)
    const colorClass = tierId === 'longevity' ? 'text-yellow-400' : 'text-green-500';
    return <Check className={`h-5 w-5 mx-auto font-bold stroke-[2.5px] ${colorClass}`} />;
  } else if (value === "✗") {
    return <X className="h-5 w-5 text-red-400 mx-auto font-bold stroke-[2.5px]" />;
  } else {
    // All text entries should be white
    return <span className="font-bold text-center block text-white">{value}</span>;
  }
};

export function PremiumComparisonTable({ 
  tierData, 
  featureCategories, 
  billingPeriod,
  onSelectTier, 
  currentTier,
  className 
}: PremiumComparisonTableProps) {

  return (
    <div className={cn('w-full', className)}>
      {/* Premium Glass Container with Elegant Boundaries */}
      <div className="glass-tier-card rounded-3xl border-4 border-white/20 shadow-2xl overflow-hidden backdrop-blur-2xl ring-2 ring-[#F15B23]/30 ring-offset-4 ring-offset-transparent">
        {/* Floating Background Elements */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-20 left-20 w-32 h-32 bg-repz-primary/5 rounded-full blur-2xl animate-pulse"></div>
          <div className="absolute bottom-20 right-20 w-40 h-40 bg-repz-complement/5 rounded-full blur-2xl animate-pulse" style={{animationDelay: '2s'}}></div>
          <div className="absolute top-1/2 left-1/2 w-24 h-24 bg-tier-performance/5 rounded-full blur-xl animate-pulse" style={{animationDelay: '4s'}}></div>
        </div>

        {/* Mobile/Tablet: Card-based layout */}
        <div className="block lg:hidden relative z-10 p-4 space-y-6">
          {featureCategories.map((category, categoryIndex) => (
            <div key={category.title} className="space-y-4">
              {/* Category Header */}
              <div className="bg-white/5 border-2 border-[#F15B23]/30 rounded-xl p-4 text-center">
                <div className="text-[#F15B23] font-bold text-lg tracking-wider uppercase glow-text">
                  {category.title}
                </div>
              </div>
              
              {/* Features for this category */}
              {category.features.map((feature, featureIndex) => (
                <div key={`${category.title}-${featureIndex}`} className="bg-gray-900/40 border border-white/10 rounded-xl p-4">
                  <div className="font-bold text-white text-center mb-4 text-lg">
                    {feature.name}
                  </div>
                  
                  {/* Grid of tier values - 2x2 on mobile, 4x1 on tablet */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {tierData.map((tier, tierIndex) => (
                      <div key={tier.id} className={cn(
                        "bg-gray-800/50 border rounded-lg p-3 text-center",
                        tier.id === 'core' && 'border-gray-400/30',
                        tier.id === 'adaptive' && 'border-pink-300/30',
                        tier.id === 'performance' && 'border-indigo-400/30',
                        tier.id === 'longevity' && 'border-amber-400/60'
                      )}>
                        <div className={cn(
                          "text-xs font-semibold mb-2",
                          tier.id === 'longevity' ? 'text-yellow-400' : 'text-gray-300'
                        )}>
                          {tier.name}
                        </div>
                        <div className="flex justify-center">
                          {renderFeatureValue(feature.values[tierIndex], tierIndex, tier.id)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* Desktop: Original table layout */}
        <div className="hidden lg:block relative z-10 overflow-x-auto border-2 border-[#F15B23]/20 rounded-2xl m-2">
          <table className="w-full min-w-[1200px] table-fixed border-collapse">
            {/* Premium Header */}
            <thead>
              <tr className="border-b-2 border-white/20">
                {/* Features Column Header - Enhanced with more prominent effect */}
                <th className="w-[240px] sticky left-0 z-10 text-center p-8 font-bold text-xl text-white bg-gradient-to-r from-gray-900/80 to-gray-800/70 border-r-2 border-white/20 relative">
                  {/* Removed "Features Comparison" title as requested */}
                  {/* Enhanced background glow */}
                  <div className="absolute inset-0 bg-gradient-to-r from-gray-900/50 to-gray-800/30 rounded-l-xl"></div>
                </th>
                
                {/* Tier Headers */}
                {tierData.map((tier) => (
                  <th key={tier.id} className={cn(
                    "w-[240px] text-center p-8 relative border-l border-white/10 border-r border-white/10",
                    // Tier-specific dark backgrounds for headers
                    tier.id === 'core' && 'bg-gradient-to-b from-gray-500/3 to-gray-600/2',
                    tier.id === 'adaptive' && 'bg-gradient-to-b from-pink-300/3 to-pink-400/2',
                    tier.id === 'performance' && 'bg-gradient-to-b from-indigo-600/3 to-purple-700/2',
                    tier.id === 'longevity' && 'bg-gradient-to-b from-amber-500/3 to-amber-600/2 border-l-2 border-r-2 border-amber-400/30'
                  )}>
                    <div className="space-y-6 relative">
                      {/* Tier Icon */}
                      <div className="flex justify-center mb-4">
                        <div className={cn(
                          'p-4 rounded-2xl shadow-lg backdrop-blur-sm',
                           tier.id === 'core' && 'bg-gray-500/20 ring-2 ring-gray-400/30',
                           tier.id === 'adaptive' && 'bg-pink-300/20 ring-2 ring-pink-300/30',
                           tier.id === 'performance' && 'bg-indigo-600/20 ring-2 ring-indigo-400/30',
                           tier.id === 'longevity' && 'bg-amber-500/20 ring-2 ring-amber-400/60'
                        )}>
                          <div className={cn(
                             tier.id === 'core' && 'text-gray-400',
                             tier.id === 'adaptive' && 'text-pink-300',
                             tier.id === 'performance' && 'text-purple-400',
                             tier.id === 'longevity' && 'text-yellow-400'
                          )}>
                            {getTierIcon(tier.id)}
                          </div>
                        </div>
                      </div>
                      
                      {/* Tier Name - White text except for Elite Concierge */}
                      <div className={cn(
                        "text-xl font-bold mb-2",
                        tier.id === 'longevity' ? 'text-yellow-400' : 'text-white'
                      )}>
                        {tier.name}
                      </div>
                      
                      {/* Description - White text except for Elite Concierge */}
                      <div className={cn(
                        "text-sm mb-4",
                        tier.id === 'longevity' ? 'text-yellow-400' : 'text-gray-300'
                      )}>
                        {tier.description}
                      </div>
                      
                      {/* Pricing - White text except for Elite Concierge */}
                      <div className="space-y-2 mb-6">
                        <div className={cn(
                          "text-4xl font-bold",
                          tier.id === 'longevity' ? 'text-yellow-400' : 'text-white'
                        )}>
                          ${getPriceForPeriod(tier, billingPeriod)}
                          <span className={cn(
                            "text-base font-normal",
                            tier.id === 'longevity' ? 'text-yellow-400' : 'text-white'
                          )}>
                            /{billingPeriod === 'annual' ? 'mo' : billingPeriod.slice(0, 2)}
                          </span>
                        </div>
                        {getSavingsForPeriod(tier, billingPeriod) > 0 && (
                          <div className="text-sm text-emerald-400 font-medium">
                            Save ${getSavingsForPeriod(tier, billingPeriod)}
                          </div>
                        )}
                      </div>
                      
                      {/* CTA Button - ENTERPRISE: All orange for consistent action hierarchy */}
                      <Button 
                        variant="primary"
                        size="lg"
                        className="w-full font-bold"
                        onClick={() => onSelectTier?.(tier.id)}
                        disabled={currentTier === tier.id}
                      >
                        {currentTier === tier.id ? 'Current Plan' : 'Join'}
                      </Button>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            
            {/* Feature Categories */}
            <tbody>
              {featureCategories.map((category, categoryIndex) => (
                <React.Fragment key={category.title}>
                  {/* Category Header - Glowing orange styling */}
                  <tr className="bg-white/5 border-y-2 border-[#F15B23]/30">
                    <td colSpan={5} className="p-6 text-center border-l-2 border-r-2 border-[#F15B23]/20">
                      <div className="flex items-center justify-center gap-3">
                        <div className="text-[#F15B23] font-bold text-lg tracking-wider uppercase relative glow-text">
                          {category.title}
                        </div>
                      </div>
                    </td>
                  </tr>
                  
                  {/* Feature Rows */}
                  {category.features.map((feature, featureIndex) => (
                    <tr key={`${category.title}-${featureIndex}`} 
                        className={cn(
                          "border-b border-white/10 hover:bg-white/5 transition-colors duration-300 group",
                          featureIndex % 2 === 0 ? "bg-gray-900/20" : "bg-gray-800/15"
                        )}>
                      {/* Feature Name - White text, centered */}
                      <td className="w-[240px] sticky left-0 z-10 p-6 font-bold text-center text-white bg-gradient-to-r from-gray-900/60 to-gray-800/50 group-hover:from-gray-900/70 border-r border-white/10">
                        <div className="flex items-center justify-center">
                          {feature.name}
                        </div>
                      </td>
                      
                      {/* Feature Values */}
                      {tierData.map((tier, tierIndex) => (
                        <td key={tier.id} className={cn(
                          "p-6 text-center transition-colors duration-300 w-[240px] border-r border-white/10",
                          // Dark tier-specific hover backgrounds
                          tier.id === 'core' && 'group-hover:bg-gray-900/30',
                          tier.id === 'adaptive' && 'group-hover:bg-pink-900/30',
                          tier.id === 'performance' && 'group-hover:bg-indigo-900/30',
                          tier.id === 'longevity' && 'group-hover:bg-amber-900/30'
                        )}>
                          {renderFeatureValue(feature.values[tierIndex], tierIndex, tier.id)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default PremiumComparisonTable;