import React, { useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/ui/atoms/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { PricingToggle, BillingPeriod } from '@/components/ui/pricing-toggle';
import { CentralizedTierCard } from '@/components/pricing/CentralizedTierCard';
import { FeatureComparisonTable } from '@/components/pricing/FeatureComparisonTable';
import { TierType, BillingCycle } from '@/constants/tiers';
import { useMobileDetection } from '@/hooks/useMobileDetection';

// CANONICAL TIER PLANS - Based on user-provided tier breakdown table
const MONTHLY_COACHING_PLANS = [
  {
    id: 'core',
    name: 'Core Program',
    monthlyPrice: 89,
    yearlyPrice: 71, // 20% discount
    description: 'Essential training foundation',
    features: [
      'Personalized Training',
      'Nutrition Plan',
      'Static/Fixed Dashboard',
      'Limited Q&A Access',
      '72 hr Response Time',
      'Science Tips'
    ],
    tier: 'core' as const
  },
  {
    id: 'adaptive',
    name: 'Adaptive Engine',
    monthlyPrice: 149,
    yearlyPrice: 119, // 20% discount
    description: 'Personalized optimization',
    features: [
      'Everything in Core PLUS:',
      'Interactive Dashboard',
      'Standard Q&A Access',
      '48 hr Response Time',
      'Weekly Check-ins & Photos',
      'Form Review',
      'Wearable Sync',
      'Sleep Optimization',
      'Auto Grocery Lists',
      'Travel Workouts',
      'Supplement Guide',
      'Research Blog Access',
      'Supplement Protocols',
      'Peptides',
      'Bioregulators',
      'Biomarker Integration',
      'Blood Work Review',
      'Recovery Guidance',
      'HRV Optimization',
      'Telegram Group',
      'Exclusive Protocols'
    ],
    tier: 'adaptive' as const
  },
  {
    id: 'performance',
    name: 'Prime Suite',
    monthlyPrice: 199,
    yearlyPrice: 159, // 20% discount
    isPopular: true, // Most Popular badge on Performance tier
    description: 'Elite athlete optimization',
    features: [
      'Everything in Adaptive PLUS:',
      'Priority Q&A Access',
      '24 hr Response Time',
      'AI Fitness Assistant',
      'AI Progress Predictors',
      'PEDs',
      'Nootropics',
      'Custom Cycling',
      'Early Access Tools'
    ],
    tier: 'performance' as const
  },
  {
    id: 'longevity',
    name: 'Elite Concierge',
    monthlyPrice: 349,
    yearlyPrice: 279, // 20% discount
    isLimited: true,
    description: 'Premium longevity protocols',
    features: [
      'Everything in Performance PLUS:',
      'Unlimited Q&A Access',
      '12 hr Response Time',
      'In-Person Training - 2×/week'
    ],
    tier: 'longevity' as const
  }
];

const ALL_FEATURES = [
  'Personalized Training',
  'Nutrition Plan',
  'Interactive Dashboard',
  'Q&A Access',
  'Response Time',
  'Weekly Check-ins & Photos',
  'Form Review',
  'Wearable Sync',
  'Sleep Optimization',
  'Auto Grocery Lists',
  'Travel Workouts',
  'Supplement Guide',
  'Research Blog Access',
  'Supplement Protocols',
  'Peptides',
  'Bioregulators',
  'Biomarker Integration',
  'Blood Work Review',
  'Recovery Guidance',
  'HRV Optimization',
  'Telegram Group',
  'Exclusive Protocols',
  'AI Fitness Assistant',
  'AI Progress Predictors',
  'PEDs',
  'Nootropics',
  'Custom Cycling',
  'Early Access Tools',
  'In-Person Training'
];

export default function MonthlyCoachingPrices() {
  const [billingPeriod, setBillingPeriod] = useState<BillingPeriod>('monthly');
  const navigate = useNavigate();
  const { isMobile } = useMobileDetection();

  const handleSelectPlan = (planId: string) => {
    console.log('Selected plan:', planId);
    // Add checkout logic here
  };

  const handleTierSelection = async (tier: TierType) => {
    console.log('Selected tier:', tier, 'billing:', billingPeriod);
    // Add Stripe checkout logic here
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate(-1)}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </Button>
          <h1 className="text-3xl font-bold text-foreground">
            Monthly Coaching Plans
          </h1>
        </div>

        {/* Pricing Toggle */}
        <PricingToggle 
          billingPeriod={billingPeriod}
          onToggle={setBillingPeriod}
          className="mb-12"
        />

        {/* Tier Cards */}
        <div className={`grid gap-6 mb-12 ${
          isMobile 
            ? 'grid-cols-1' 
            : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4'
        }`}>
          {MONTHLY_COACHING_PLANS.map((plan) => (
            <CentralizedTierCard
              key={plan.id}
              tier={plan.tier}
              billingCycle={billingPeriod === 'annual' ? 'annual' : 'monthly'}
              features={plan.features}
              monthlyPrice={plan.monthlyPrice}
              yearlyPrice={plan.yearlyPrice}
              description={plan.description}
              isPopular={plan.isPopular}
              onSelect={handleTierSelection}
            />
          ))}
        </div>

        {/* Feature Comparison Table */}
        <FeatureComparisonTable 
          tiers={MONTHLY_COACHING_PLANS.map(plan => ({
            id: plan.id,
            name: plan.name,
            features: plan.features
          }))}
          allFeatures={ALL_FEATURES}
        />
      </div>
    </div>
  );
}