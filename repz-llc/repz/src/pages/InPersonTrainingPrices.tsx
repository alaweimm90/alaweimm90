import React, { useState } from 'react';
import { RepzCard } from '@/components/ui/repz-card';
import { RepzButton } from '@/components/ui/repz-button';
import { PricingToggle, BillingPeriod } from '@/components/ui/pricing-toggle';
import { ArrowLeft } from 'lucide-react';
import { UnifiedTierCard } from '@/components/ui/tier-card';
import { useNavigate } from 'react-router-dom';
import { CalendlyBooking } from '@/components/booking/CalendlyBooking';

const IN_PERSON_TRAINING_PLANS = [
  {
    id: 'personal-training',
    name: 'Personal Training',
    monthlyPrice: 200,
    yearlyPrice: 1920, // 20% discount
    description: 'Individual training sessions',
    features: [
      '1-on-1 personal training sessions',
      'Customized workout plans',
      'Form correction and technique coaching',
      'Progress tracking and assessments',
      'Nutritional guidance',
      'Flexible scheduling',
      'Equipment instruction',
      'Goal-specific programming'
    ],
    tier: 'core' as const
  },
  {
    id: 'small-group',
    name: 'Small Group Training',
    monthlyPrice: 150,
    yearlyPrice: 1440, // 20% discount
    description: '2-4 person group sessions',
    features: [
      'Small group training (2-4 people)',
      'Shared workout experiences',
      'Individual attention within group setting',
      'Cost-effective training option',
      'Motivational group environment',
      'Customized group programs',
      'Progress tracking for each member',
      'Team building exercises'
    ],
    tier: 'adaptive' as const,
    isPopular: true
  },
  {
    id: 'athletic-performance',
    name: 'Athletic Performance',
    monthlyPrice: 300,
    yearlyPrice: 2880, // 20% discount
    description: 'Sport-specific training',
    features: [
      'Sport-specific training programs',
      'Performance testing and analysis',
      'Agility and speed development',
      'Strength and conditioning',
      'Injury prevention protocols',
      'Recovery and regeneration techniques',
      'Mental performance coaching',
      'Competition preparation'
    ],
    tier: 'performance' as const
  },
  {
    id: 'executive-wellness',
    name: 'Executive Wellness',
    monthlyPrice: 500,
    yearlyPrice: 4800, // 20% discount
    description: 'Premium concierge fitness',
    features: [
      'Concierge-level personal training',
      'Flexible location options (home, office, gym)',
      'Comprehensive health assessments',
      'Stress management and wellness coaching',
      'Executive health optimization',
      'Meal planning and prep coordination',
      'Recovery and longevity protocols',
      '24/7 health and fitness support'
    ],
    tier: 'longevity' as const,
    isLimited: true
  }
];

export default function InPersonTrainingPrices() {
  const [billingPeriod, setBillingPeriod] = useState<BillingPeriod>('monthly');
  const navigate = useNavigate();

  const handleSelectPlan = (planId: string) => {
    console.log('Selected plan:', planId);
    // Handle plan selection logic here
  };

  return (
    <div className="min-h-screen bg-repz-black text-text-primary">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <RepzButton
            variant="tier"
            size="sm"
            onClick={() => navigate('/')}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </RepzButton>
        </div>

        {/* Title */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-repz-orange mb-4">
            In-Person Training Plans
          </h1>
          <p className="text-text-secondary text-lg max-w-2xl mx-auto">
            Compare our in-person training options designed for hands-on coaching and personalized attention.
          </p>
        </div>

        {/* Pricing Toggle */}
        <PricingToggle 
          billingPeriod={billingPeriod}
          onToggle={setBillingPeriod}
        />

        {/* Plans Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-12">
          {IN_PERSON_TRAINING_PLANS.map((plan, index) => (
            <UnifiedTierCard
              key={plan.id}
              tier={plan.tier}
              billingCycle={billingPeriod === 'monthly' ? 'monthly' : 'annual'}
              variant={plan.isPopular ? 'detailed' : 'comparison'}
              onSelect={() => handleSelectPlan(plan.id)}
              showFeatures={true}
            />
          ))}
        </div>

        {/* Feature Comparison Table */}
        <RepzCard variant="elevated" className="p-8">
          <h2 className="text-2xl font-bold text-center mb-8">Detailed Feature Comparison</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-4 px-4 text-text-secondary">Features</th>
                  <th className="text-center py-4 px-4 text-tier-core">Personal</th>
                  <th className="text-center py-4 px-4 text-tier-adaptive">Small Group</th>
                  <th className="text-center py-4 px-4 text-tier-performance">Athletic</th>
                  <th className="text-center py-4 px-4 text-tier-longevity">Executive</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { feature: 'Individual attention', personal: true, smallGroup: true, athletic: true, executive: true },
                  { feature: 'Group motivation', personal: false, smallGroup: true, athletic: true, executive: false },
                  { feature: 'Sport-specific training', personal: false, smallGroup: false, athletic: true, executive: false },
                  { feature: 'Location flexibility', personal: 'Limited', smallGroup: 'Limited', athletic: 'Gym only', executive: 'Unlimited' },
                  { feature: 'Session duration', personal: '60 min', smallGroup: '60 min', athletic: '90 min', executive: 'Flexible' },
                  { feature: 'Nutritional guidance', personal: true, smallGroup: true, athletic: true, executive: true },
                  { feature: 'Recovery protocols', personal: false, smallGroup: false, athletic: true, executive: true },
                  { feature: 'Performance testing', personal: false, smallGroup: false, athletic: true, executive: true },
                  { feature: '24/7 support', personal: false, smallGroup: false, athletic: false, executive: true },
                  { feature: 'Concierge services', personal: false, smallGroup: false, athletic: false, executive: true }
                ].map((row, index) => (
                  <tr key={index} className="border-b border-white/5">
                    <td className="py-3 px-4 text-text-primary">{row.feature}</td>
                    <td className="py-3 px-4 text-center">
                      {typeof row.personal === 'boolean' ? (
                        row.personal ? <span className="text-success">✓</span> : <span className="text-red-500">✗</span>
                      ) : (
                        <span className="text-text-secondary text-sm">{row.personal}</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-center">
                      {typeof row.smallGroup === 'boolean' ? (
                        row.smallGroup ? <span className="text-success">✓</span> : <span className="text-red-500">✗</span>
                      ) : (
                        <span className="text-text-secondary text-sm">{row.smallGroup}</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-center">
                      {typeof row.athletic === 'boolean' ? (
                        row.athletic ? <span className="text-success">✓</span> : <span className="text-red-500">✗</span>
                      ) : (
                        <span className="text-text-secondary text-sm">{row.athletic}</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-center">
                      {typeof row.executive === 'boolean' ? (
                        row.executive ? <span className="text-success">✓</span> : <span className="text-red-500">✗</span>
                      ) : (
                        <span className="text-text-secondary text-sm">{row.executive}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </RepzCard>
      </div>
    </div>
  );
}