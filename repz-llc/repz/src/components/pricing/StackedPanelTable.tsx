import React from 'react';
import { Card } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Check, X, Zap, Award, Crown, Star } from 'lucide-react';
import { TIER_CONFIGS } from '@/constants/tiers';
import { cn } from '@/lib/utils';
import { AnimatedSection } from '@/components/ui/animated-section';

interface StackedPanelTableProps {
  onTierSelect?: (tierId: string) => void;
  className?: string;
}

export const StackedPanelTable: React.FC<StackedPanelTableProps> = ({
  onTierSelect,
  className = ""
}) => {
  // Feature comparison data with tier availability
  const features = [
    { 
      name: "Personalized training program", 
      core: true, 
      adaptive: true, 
      performance: true, 
      longevity: true,
      category: "Core Platform & Program"
    },
    { 
      name: "Macro-based nutrition plan", 
      core: true, 
      adaptive: true, 
      performance: true, 
      longevity: true,
      category: "Core Platform & Program"
    },
    { 
      name: "Q&A access", 
      core: "72hr response", 
      adaptive: "48hr response", 
      performance: "24hr response", 
      longevity: "12hr response",
      category: "Coach Access & Support"
    },
    { 
      name: "Weekly check-ins & photo reviews", 
      core: false, 
      adaptive: true, 
      performance: true, 
      longevity: true,
      category: "Progress Tracking & Analysis"
    },
    { 
      name: "Wearable device integration", 
      core: false, 
      adaptive: true, 
      performance: true, 
      longevity: true,
      category: "Progress Tracking & Analysis"
    },
    { 
      name: "AI fitness assistant & predictions", 
      core: false, 
      adaptive: false, 
      performance: true, 
      longevity: true,
      category: "Progress Tracking & Analysis"
    },
    { 
      name: "Biomarker integration", 
      core: false, 
      adaptive: false, 
      performance: true, 
      longevity: true,
      category: "Health Analytics"
    },
    { 
      name: "Peptides & PEDs protocols", 
      core: false, 
      adaptive: false, 
      performance: true, 
      longevity: true,
      category: "Biohacking & Advanced Supplementation"
    },
    { 
      name: "Private Telegram community", 
      core: false, 
      adaptive: false, 
      performance: true, 
      longevity: true,
      category: "Community & Exclusive Access"
    },
    { 
      name: "In-person training (2x/week)", 
      core: false, 
      adaptive: false, 
      performance: false, 
      longevity: true,
      category: "Premium Services"
    },
    { 
      name: "Dedicated account manager", 
      core: false, 
      adaptive: false, 
      performance: false, 
      longevity: true,
      category: "Premium Services"
    },
    { 
      name: "HRV optimization", 
      core: false, 
      adaptive: false, 
      performance: false, 
      longevity: true,
      category: "Health Analytics"
    }
  ];

  const renderFeatureValue = (value: boolean | string, isLongevity: boolean = false) => {
    if (typeof value === 'string') {
      return (
        <div className="flex items-center justify-center">
          <span className={cn(
            "text-xs px-2 py-1 rounded-full",
            isLongevity 
              ? "text-yellow-200 bg-yellow-500/20 border border-yellow-500/30" 
              : "text-gray-600 bg-gray-100"
          )}>
            {value}
          </span>
        </div>
      );
    }
    
    return (
      <div className="flex items-center justify-center">
        {value ? (
          <Check className={cn(
            "w-5 h-5",
            isLongevity ? "text-yellow-400" : "text-green-500"
          )} />
        ) : (
          <X className={cn(
            "w-4 h-4",
            isLongevity ? "text-gray-500" : "text-gray-300"
          )} />
        )}
      </div>
    );
  };

  return (
    <AnimatedSection animation="fade-in">
      <div className={cn("w-full max-w-7xl mx-auto", className)}>
        <Card className="overflow-hidden shadow-xl border-0 bg-white">
          {/* Header Row - Tier Information */}
          <div className="grid grid-cols-5 border-b border-gray-200">
            {/* Features Header */}
            <div className="p-6 flex items-center justify-center border-r border-gray-200 bg-gradient-to-br from-gray-800 to-gray-900">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">
                  <Check className="w-4 h-4 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-white">Features</h3>
              </div>
            </div>
            
            {/* Tier Headers */}
            {TIER_CONFIGS.map((tier, index) => {
              const getTierIcon = (iconName: string) => {
                switch (iconName) {
                  case 'Zap': return Zap;
                  case 'Award': return Award;
                  case 'Crown': return Crown;
                  case 'Star': return Star;
                  default: return Zap;
                }
              };
              const TierIcon = getTierIcon(tier.icon);
              const isPopular = tier.isPopular;
              const isPremium = tier.id === 'longevity';
              
              return (
                <div 
                  key={tier.id}
                  className={cn(
                    "relative p-6 border-r border-gray-200 last:border-r-0",
                    "flex flex-col items-center justify-center text-center min-h-[240px]",
                    isPopular ? "bg-gradient-to-b from-orange-50 to-orange-100 ring-2 ring-orange-200" : "",
                    isPremium ? "bg-gradient-to-b from-black to-gray-900" : "",
                    !isPopular && !isPremium ? "bg-gradient-to-b from-gray-50 to-gray-100" : ""
                  )}
                >
                  {/* Popular Badge - positioned above tier cards */}
                  {isPopular && (
                    <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 z-10">
                      <Badge className="bg-orange-500 text-white px-4 py-2 text-sm font-semibold shadow-lg rounded-full">
                        MOST POPULAR
                      </Badge>
                    </div>
                  )}
                  
                  {/* Premium Badge - positioned above tier cards */}
                  {isPremium && (
                    <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 z-10">
                      <Badge className="bg-gradient-to-r from-yellow-500 to-yellow-600 text-black px-4 py-2 text-sm font-semibold shadow-lg rounded-full">
                        PREMIUM TIER
                      </Badge>
                    </div>
                  )}
                  
                  <div className="flex flex-col items-center justify-center space-y-4">
                    {/* Icon - vertically centered, doubled size */}
                    <div 
                      className={cn(
                        "w-16 h-16 rounded-full flex items-center justify-center",
                        isPremium ? "bg-yellow-500/20" : ""
                      )}
                      style={{ 
                        backgroundColor: !isPremium ? tier.colors.primary + '20' : undefined 
                      }}
                    >
                      <TierIcon 
                        className="w-8 h-8" 
                        style={{ 
                          color: isPremium ? '#FFD700' : tier.colors.primary 
                        }} 
                      />
                    </div>
                    
                    {/* Tier Name & Subname - vertically centered */}
                    <div className="space-y-1">
                      <h3 
                        className="text-xl font-bold"
                        style={{ 
                          color: isPremium ? '#FFD700' : tier.colors.primary 
                        }}
                      >
                        {tier.displayName}
                      </h3>
                      <p className={cn(
                        "text-sm font-medium",
                        isPremium ? "text-yellow-200" : "text-gray-600"
                      )}>
                        {tier.description}
                      </p>
                    </div>
                    
                    {/* Price - vertically centered */}
                    <div className="space-y-1">
                      <div 
                        className="text-3xl font-bold"
                        style={{ 
                          color: isPremium ? '#FFD700' : tier.colors.primary 
                        }}
                      >
                        {tier.priceDisplay}
                      </div>
                      <div className={cn(
                        "text-sm font-medium",
                        isPremium ? "text-yellow-200" : "text-gray-500"
                      )}>
                        per month
                      </div>
                    </div>
                    
                    {/* CTA Button */}
                    <Button
                      onClick={() => onTierSelect?.(tier.id)}
                      className={cn(
                        "w-full mt-4 font-semibold transition-all duration-200",
                        isPopular 
                          ? "bg-orange-500 hover:bg-orange-600 text-white" 
                          : isPremium
                          ? "bg-gradient-to-r from-yellow-500 to-yellow-600 hover:from-yellow-600 hover:to-yellow-700 text-black"
                          : "hover:scale-105"
                      )}
                      style={{
                        backgroundColor: !isPopular && !isPremium ? tier.colors.primary : undefined,
                        borderColor: isPremium ? '#FFD700' : tier.colors.primary,
                        color: !isPopular && !isPremium ? 'white' : undefined
                      }}
                    >
                      Choose {tier.displayName.split(' ')[0]}
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Feature Rows */}
          <div>
            {features.map((feature, index) => (
              <div 
                key={feature.name}
                className={cn(
                  "grid grid-cols-5 border-b border-gray-100 hover:bg-gray-50 transition-colors",
                  index % 2 === 0 ? "bg-white" : "bg-gray-50/50"
                )}
              >
                {/* Feature Name */}
                <div className="p-4 border-r border-gray-200 flex items-center">
                  <div>
                    <span className="font-medium text-gray-900">{feature.name}</span>
                    <div className="text-xs text-gray-500 mt-1">{feature.category}</div>
                  </div>
                </div>
                
                {/* Feature Availability for Each Tier */}
                <div className="p-4 border-r border-gray-200 flex items-center justify-center">
                  {renderFeatureValue(feature.core)}
                </div>
                <div className={cn(
                  "p-4 border-r border-gray-200 flex items-center justify-center",
                  "bg-orange-50/30"
                )}>
                  {renderFeatureValue(feature.adaptive)}
                </div>
                <div className="p-4 border-r border-gray-200 flex items-center justify-center">
                  {renderFeatureValue(feature.performance)}
                </div>
                <div className={cn(
                  "p-4 flex items-center justify-center",
                  "bg-gradient-to-b from-black/90 to-gray-900/90"
                )}>
                  {renderFeatureValue(feature.longevity, true)}
                </div>
              </div>
            ))}
          </div>

          {/* Footer CTA Row */}
          <div className="grid grid-cols-5 bg-gradient-to-r from-gray-50 to-gray-100 p-6">
            <div className="border-r border-gray-200 flex items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900">
              <p className="text-sm font-semibold text-white">Ready to start?</p>
            </div>
            
            {TIER_CONFIGS.map((tier) => {
              const isLongevity = tier.id === 'longevity';
              return (
                <div 
                  key={`footer-${tier.id}`}
                  className={cn(
                    "border-r border-gray-200 last:border-r-0 flex items-center justify-center px-4",
                    isLongevity ? "bg-gradient-to-b from-black to-gray-900" : ""
                  )}
                >
                  <Button
                    onClick={() => onTierSelect?.(tier.id)}
                    variant="outline"
                    className={cn(
                      "w-full font-semibold transition-all duration-200 hover:scale-105",
                      tier.popular ? "border-orange-500 text-orange-600 hover:bg-orange-50" : "",
                      isLongevity ? "border-yellow-500 text-yellow-400 hover:bg-yellow-500/10" : ""
                    )}
                    style={{
                      borderColor: isLongevity ? '#FFD700' : tier.colors.primary,
                      color: isLongevity ? '#FFD700' : tier.colors.primary
                    }}
                  >
                    Get Started
                  </Button>
                </div>
              );
            })}
          </div>
        </Card>
      </div>
    </AnimatedSection>
  );
};

export default StackedPanelTable;