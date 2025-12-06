import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Button } from "@/components/ui/unified-button";
import { Badge } from "@/ui/atoms/Badge";
import { Check, X, Star, Zap, Crown, LayoutGrid, List, Sprout, TrendingUp } from 'lucide-react';
import { useMediaQuery } from '@/hooks/useMediaQuery';
import { cn } from '@/lib/utils';
import { PremiumComparisonTable } from '@/components/ui/premium-comparison-table';
import { MonthlyCoachingCard } from '@/components/ui/unified-pricing-card';
import type { BillingPeriod } from '@/components/ui/pricing-toggle';
import type { TierType, BadgeType } from '@/components/ui/unified-pricing-card';

interface TierFeature {
  name: string;
  included: boolean;
  highlight?: boolean;
}

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
  gradient: string;
  buttonText: string;
  testId?: string;
}

interface EnhancedTierDisplayProps {
  billingPeriod: BillingPeriod;
  onSelectTier?: (tierId: string) => void;
  currentTier?: string;
  className?: string;
}

const tierData: TierData[] = [
  {
    id: "core",
    name: "Core Program", 
    description: "Perfect for fitness beginners ready to transform",
    price: { monthly: 89, quarterly: 85, semiannual: 80, annual: 71 },
    savings: { quarterly: 178, semiannual: 534, annual: 1068 },
    icon: <Sprout className="h-5 w-5" />,
    gradient: "from-blue-500 to-blue-600",
    buttonText: "Join",
    testId: "tier-card-core",
    features: [
      { name: "Everything you need to start", included: true },
      { name: "Personalized training & nutrition", included: true },
      { name: "Science-based guidance", included: true },
      { name: "72hr coach support", included: true }
    ]
  },
  {
    id: "adaptive",
    name: "Adaptive Engine",
    description: "For serious athletes who want smart tracking",
    price: { monthly: 149, quarterly: 142, semiannual: 134, annual: 119 },
    savings: { quarterly: 298, semiannual: 894, annual: 1788 },
    icon: <TrendingUp className="h-5 w-5" />,
    gradient: "from-orange-500 to-orange-600",
    buttonText: "Join",
    testId: "tier-card-adaptive",
    features: [
      { name: "Interactive AI dashboard", included: true, highlight: true },
      { name: "Weekly photo check-ins", included: true, highlight: true },
      { name: "Form reviews & wearable sync", included: true, highlight: true },
      { name: "Peptides & community access", included: true, highlight: true },
      { name: "48hr priority support", included: true }
    ]
  },
  {
    id: "performance",
    name: "Prime Suite",
    description: "Elite optimization with AI + enhancement protocols",
    price: { monthly: 229, quarterly: 218, semiannual: 206, annual: 183 },
    savings: { quarterly: 458, semiannual: 1374, annual: 2748 },
    popular: true,
    icon: <Zap className="h-5 w-5" />,
    gradient: "from-purple-500 to-purple-600",
    buttonText: "Join",
    testId: "tier-card-performance",
    features: [
      { name: "AI Fitness Assistant", included: true, highlight: true },
      { name: "PEDs & nootropics protocols", included: true, highlight: true },
      { name: "Priority 24hr support", included: true, highlight: true },
      { name: "Everything from Adaptive +", included: true },
      { name: "Advanced enhancement stack", included: true, highlight: true }
    ]
  },
  {
    id: "longevity",
    name: "Elite Concierge",
    description: "Ultimate luxury with in-person training",
    price: { monthly: 349, quarterly: 332, semiannual: 314, annual: 279 },
    savings: { quarterly: 698, semiannual: 2094, annual: 4188 },
    premium: true,
    icon: <Crown className="h-5 w-5" />,
    gradient: "from-yellow-500 to-yellow-600",
    buttonText: "Join",
    testId: "tier-card-longevity",
    features: [
      { name: "In-person training 2x/week", included: true, highlight: true },
      { name: "12hr VIP support", included: true, highlight: true },
      { name: "Complete protocol access", included: true, highlight: true },
      { name: "Everything from Performance +", included: true },
      { name: "White-glove concierge service", included: true, highlight: true }
    ]
  }
];

export function EnhancedTierDisplay({ 
  billingPeriod,
  onSelectTier, 
  currentTier,
  className 
}: EnhancedTierDisplayProps) {
  const isMobile = useMediaQuery('(max-width: 768px)');
  const [viewMode, setViewMode] = useState<'cards' | 'comparison'>('cards');

  const getPriceForPeriod = (tier: TierData, period: BillingPeriod) => {
    const priceMap = { monthly: tier.price.monthly, quarterly: tier.price.quarterly, 'semi-annual': tier.price.semiannual, annual: tier.price.annual };
    return priceMap[period];
  };

  const getSavingsForPeriod = (tier: TierData, period: BillingPeriod) => {
    if (period === 'monthly') return 0;
    const savingsMap = { quarterly: tier.savings.quarterly, 'semi-annual': tier.savings.semiannual, annual: tier.savings.annual };
    return savingsMap[period] || 0;
  };

  const getCardClasses = (tier: TierData) => {
    const baseClasses = "relative rounded-2xl backdrop-blur-xl transition-all duration-500 hover:shadow-2xl overflow-hidden group border-2 bg-black w-full h-full min-h-[480px] flex flex-col";
    
    // ENTERPRISE: Tier-specific colored outlines with enhanced glowing effects
    const tierStyles = {
      "core": "border-gray-400/60 hover:border-gray-400/80 hover:shadow-gray-400/25 hover:shadow-2xl",
      "adaptive": "border-pink-300/60 hover:border-pink-300/80 hover:shadow-pink-300/25 hover:shadow-2xl",
      "performance": "border-indigo-600/60 hover:border-indigo-600/80 hover:shadow-indigo-600/25 hover:shadow-2xl",
      "longevity": "border-amber-400/90 shadow-[0_0_20px_rgba(251,191,36,0.3),0_0_40px_rgba(251,191,36,0.1),inset_0_0_20px_rgba(251,191,36,0.05)] hover:shadow-[0_0_30px_rgba(251,191,36,0.4),0_0_60px_rgba(251,191,36,0.2),inset_0_0_30px_rgba(251,191,36,0.1)]"
    };

    return `${baseClasses} ${tierStyles[tier.id as keyof typeof tierStyles] || tierStyles.core}`;
  };

  // ENTERPRISE: ALL CTA buttons use primary orange for consistent action hierarchy
  const getTierButtonVariant = (tierId: string) => {
    return 'primary'; // All tier buttons use primary (orange) variant
  };

  const getTextClasses = (tier: TierData) => {
    return tier.id === "longevity" ? "text-amber-400" : "text-white";
  };

  const getAnimatedBackground = (tier: TierData) => {
    // Enhanced animated backgrounds without pulse animation
    const bgStyles = {
      "core": "bg-gradient-to-br from-gray-500/3 to-gray-600/1",
      "adaptive": "bg-gradient-to-br from-pink-300/3 to-pink-400/1", 
      "performance": "bg-gradient-to-br from-indigo-600/3 to-purple-700/1",
      "longevity": "bg-gradient-to-br from-amber-500/8 via-yellow-400/6 to-amber-600/4"
    };
    
    return `absolute inset-0 opacity-40 ${bgStyles[tier.id as keyof typeof bgStyles] || bgStyles.core}`;
  };

  // Helper functions for unified card architecture
  const getBadgeForTier = (tierId: string): BadgeType | undefined => {
    const tier = tierData.find(t => t.id === tierId);
    if (tier?.popular) return 'popular';
    if (tier?.premium) return 'limited';
    return undefined;
  };

  const getTierType = (tierId: string): TierType => {
    const tierMap: Record<string, TierType> = {
      'core': 'core',
      'adaptive': 'adaptive', 
      'performance': 'performance',
      'longevity': 'longevity'
    };
    return tierMap[tierId] || 'core';
  };

  const renderCards = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-8xl mx-auto place-items-center">
      {tierData.map((tier) => (
        <MonthlyCoachingCard
          key={tier.id}
          title={tier.name}
          subtitle={tier.description}
          price={billingPeriod === 'annual' 
            ? `$${getPriceForPeriod(tier, billingPeriod)}/mo` 
            : `$${getPriceForPeriod(tier, billingPeriod)}/mo`
          }
          priceSubtext={billingPeriod === 'annual' 
            ? `Save $${getSavingsForPeriod(tier, billingPeriod)}/year` 
            : billingPeriod === 'monthly' ? 'Billed monthly' : `Save $${getSavingsForPeriod(tier, billingPeriod)}/year`
          }
          priceSubtextColor="text-green-400"
          description={undefined}
          tier={getTierType(tier.id)}
          features={tier.features.map(feature => ({
            text: feature.name,
            included: true,
            highlight: feature.highlight,
            iconColor: tier.id === 'longevity' ? 'text-amber-400' : 'text-green-400'
          }))}
          badge={getBadgeForTier(tier.id)}
          badgePosition="top-right"
          buttonText={currentTier === tier.id ? "Current Plan" : tier.buttonText}
          buttonVariant="primary"
          onClick={() => onSelectTier?.(tier.id)}
          isSelected={currentTier === tier.id}
          className="w-full max-w-[360px]"
          comingSoon={true}
          comingSoonMessage="New, modern dashboards launching soon for all subscription plans."
        />
      ))}
    </div>
  );

  // ENTERPRISE: Feature categories for comparison table
  const featureCategories = [
    {
      title: "CORE PLATFORM",
      features: [
        { name: "Personalized Training", values: ["✓", "✓", "✓", "✓"] },
        { name: "Nutrition Plan", values: ["✓", "✓", "✓", "✓"] },
        { name: "Dashboard Program", values: ["Static/fixed", "Interactive/adjustable", "Interactive/adjustable", "Interactive/adjustable"] },
        { name: "Science-based Tips", values: ["✓", "✓", "✓", "✓"] },
        { name: "Supplements Guide", values: ["✓", "✓", "✓", "✓"] }
      ]
    },
    {
      title: "COACH ACCESS", 
      features: [
        { name: "Q&A Access", values: ["Limited", "Standard", "Priority", "Unlimited"] },
        { name: "Response Time", values: ["72hr", "48hr", "24hr", "12hr"] }
      ]
    },
    {
      title: "PROGRESS TRACKING",
      features: [
        { name: "Weekly Check-ins & Photos", values: ["✗", "✓", "✓", "✓"] },
        { name: "Form Review", values: ["✗", "✓", "✓", "✓"] },
        { name: "Wearable Sync", values: ["✗", "✓", "✓", "✓"] },
        { name: "Sleep Optimization", values: ["✗", "✓", "✓", "✓"] },
        { name: "AI Fitness Assistant", values: ["✗", "✗", "✓", "✓"] },
        { name: "AI Progress Predictors", values: ["✗", "✗", "✓", "✓"] }
      ]
    },
    {
      title: "CONVENIENCE",
      features: [
        { name: "Auto Grocery Lists", values: ["✗", "✓", "✓", "✓"] },
        { name: "Travel Workouts", values: ["✗", "✓", "✓", "✓"] }
      ]
    },
    {
      title: "SUPPLEMENTATION", 
      features: [
        { name: "Supplement Protocols", values: ["✗", "✓", "✓", "✓"] },
        { name: "Peptides", values: ["✗", "✓", "✓", "✓"] },
        { name: "PEDs", values: ["✗", "✗", "✓", "✓"] },
        { name: "Nootropics", values: ["✗", "✗", "✓", "✓"] },
        { name: "Bioregulators", values: ["✗", "✓", "✓", "✓"] }
      ]
    },
    {
      title: "ANALYTICS",
      features: [
        { name: "Biomarker Integration", values: ["✗", "✓", "✓", "✓"] },
        { name: "Blood Work Review", values: ["✗", "✓", "✓", "✓"] },
        { name: "Recovery Guidance", values: ["✗", "✓", "✓", "✓"] },
        { name: "HRV Optimization", values: ["✗", "✓", "✓", "✓"] }
      ]
    },
    {
      title: "COMMUNITY & ACCESS",
      features: [
        { name: "Telegram Group", values: ["✗", "✓", "✓", "✓"] },
        { name: "Exclusive Protocols", values: ["✗", "✓", "✓", "✓"] },
        { name: "Early Access Tools", values: ["✗", "✗", "✓", "✓"] }
      ]
    },
    {
      title: "PREMIUM SERVICES",
      features: [
        { name: "In-Person Training", values: ["✗", "✗", "✗", "2x/week"] }
      ]
    }
  ];

  return (
    <div className={cn('w-full', className)}>
      {/* View Toggle - Using Enterprise Button System */}
      <div className="flex justify-center mb-8">
        <div className="bg-gradient-to-r from-gray-800/60 to-gray-700/40 border border-gray-600/40 rounded-lg p-1 flex backdrop-blur-sm">
          <Button
            variant={viewMode === 'cards' ? 'primary' : 'ghost'}
            size="sm"
            onClick={() => setViewMode('cards')}
            className="flex items-center gap-2"
            style={{ 
              color: viewMode === 'cards' ? '#FFFFFF' : '#F29C00',
              borderColor: viewMode === 'cards' ? '#F15B23' : '#F29C00'
            }}
          >
            <LayoutGrid className="h-4 w-4" />
            Cards View
          </Button>
          <Button
            variant={viewMode === 'comparison' ? 'primary' : 'ghost'}
            size="sm"
            onClick={() => setViewMode('comparison')}
            className="flex items-center gap-2"
            style={{ 
              color: viewMode === 'comparison' ? '#FFFFFF' : '#F29C00',
              borderColor: viewMode === 'comparison' ? '#F15B23' : '#F29C00'
            }}
          >
            <List className="h-4 w-4" />
            Compare All
          </Button>
        </div>
      </div>

      {viewMode === 'cards' ? renderCards() : (
        <PremiumComparisonTable 
          tierData={tierData}
          featureCategories={featureCategories}
          billingPeriod={billingPeriod}
          onSelectTier={onSelectTier}
          currentTier={currentTier}
        />
      )}
    </div>
  );
}