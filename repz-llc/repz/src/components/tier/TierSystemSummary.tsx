// TIER SYSTEM IMPLEMENTATION SUMMARY
// Complete overview of the canonical tier structure

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { 
  CheckCircle, 
  DollarSign, 
  Users, 
  Zap, 
  Shield, 
  BarChart3,
  Calendar,
  CreditCard
} from 'lucide-react';
import { UnifiedTierCard } from '@/components/ui/unified-tier-card';
import { TierType, BillingCycle, getTierConfig } from '@/constants/tiers';
import { TIER_FEATURES, hasFeatureAccess } from '@/constants/featureMatrix';

interface TierSystemSummaryProps {
  onDemoCheckout?: (tier: TierType, billing: BillingCycle) => void;
}

export const TierSystemSummary: React.FC<TierSystemSummaryProps> = ({
  onDemoCheckout
}) => {
  const [activeTab, setActiveTab] = useState('overview');

  const handleDemoCheckout = (tier: TierType, billing: BillingCycle) => {
    console.log(`Demo checkout: ${tier} - ${billing}`);
    if (onDemoCheckout) {
      onDemoCheckout(tier, billing);
    } else {
      alert(`Demo: Would checkout ${tier} tier with ${billing} billing`);
    }
  };

  const stats = [
    { label: 'Tier Levels', value: '4', icon: BarChart3 },
    { label: 'Billing Cycles', value: '4', icon: Calendar },
    { label: 'Features', value: '20+', icon: Zap },
    { label: 'Price Range', value: '$89-$349', icon: DollarSign }
  ];

  const features = Object.keys(TIER_FEATURES);
  const tiers: TierType[] = ['core', 'adaptive', 'performance', 'longevity'];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-orange-500 to-orange-600 bg-clip-text text-transparent mb-4">
          Canonical Tier System
        </h1>
        <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
          Complete implementation of the REPZ coaching tier structure with conversion-optimized pricing, 
          comprehensive feature matrix, and seamless checkout integration.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat, idx) => (
          <Card key={idx} className="text-center">
            <CardContent className="pt-6">
              <div className="flex items-center justify-center mb-2">
                <stat.icon className="w-8 h-8 text-orange-500" />
              </div>
              <div className="text-2xl font-bold">{stat.value}</div>
              <div className="text-sm text-muted-foreground">{stat.label}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="pricing">Pricing Cards</TabsTrigger>
          <TabsTrigger value="features">Feature Matrix</TabsTrigger>
          <TabsTrigger value="integration">Integration</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="w-5 h-5" />
                Tier Structure Overview
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {tiers.map((tier) => {
                  const config = getTierConfig(tier);
                  return (
                    <div key={tier} className="p-4 border rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <div className={`w-3 h-3 rounded-full bg-${tier === 'core' ? 'blue' : tier === 'adaptive' ? 'orange' : tier === 'performance' ? 'purple' : 'yellow'}-500`} />
                        <h3 className="font-semibold">{config.displayName}</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-3">
                        {config.description}
                      </p>
                      <div className="space-y-1 text-xs">
                        <div>Monthly: ${config.monthly_price}</div>
                        <div>Annual: ${Math.round(config.charm_pricing.annual / 12)}/mo</div>
                        <div>Response: {TIER_FEATURES[tier].response_time_hours}hr</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Implementation Highlights</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Canonical Structure
                  </h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• Single source of truth in constants/</li>
                    <li>• Consistent naming across all components</li>
                    <li>• No legacy tier references</li>
                    <li>• Standardized feature matrix</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Conversion Optimization
                  </h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• Psychology-based pricing ($89, $149, $229, $349)</li>
                    <li>• Anchoring with annual discounts</li>
                    <li>• Social proof and urgency</li>
                    <li>• Clear value propositions</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Technical Excellence
                  </h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• TypeScript type safety</li>
                    <li>• Responsive design system</li>
                    <li>• Secure Stripe integration</li>
                    <li>• Comprehensive testing</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-3 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Seamless UX
                  </h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• One-click billing cycle switching</li>
                    <li>• Real-time pricing calculations</li>
                    <li>• Beautiful success flows</li>
                    <li>• Mobile-optimized experience</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="pricing" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Live Pricing Cards</CardTitle>
              <p className="text-sm text-muted-foreground">
                Interactive tier cards with all billing cycles and conversion features
              </p>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {tiers.map((tier) => (
                  <UnifiedTierCard
                    key={tier}
                    tier={tier}
                    variant="detailed"
                    onSelect={(selectedTier) => handleDemoCheckout(selectedTier, 'monthly')}
                  />
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="features" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Complete Feature Matrix</CardTitle>
              <p className="text-sm text-muted-foreground">
                All features across all tiers with clear access control
              </p>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Feature</th>
                      {tiers.map((tier) => (
                        <th key={tier} className="text-center p-2 capitalize">
                          {tier}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {features.map((feature) => (
                      <tr key={feature} className="border-b">
                        <td className="p-2 font-medium">
                          {feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </td>
                        {tiers.map((tier) => (
                          <td key={tier} className="text-center p-2">
                            {hasFeatureAccess(tier, feature as keyof typeof TIER_FEATURES) ? (
                              <CheckCircle className="w-4 h-4 text-green-500 mx-auto" />
                            ) : (
                              <div className="w-4 h-4 mx-auto rounded-full bg-gray-200" />
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="integration" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CreditCard className="w-5 h-5" />
                  Stripe Integration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  <span className="text-sm">Edge function: create-comprehensive-checkout</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  <span className="text-sm">Support for all billing cycles</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  <span className="text-sm">Secure price configuration</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  <span className="text-sm">Comprehensive error handling</span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="w-5 h-5" />
                  User Experience
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  <span className="text-sm">Conversion-optimized design</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  <span className="text-sm">Mobile-responsive layout</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  <span className="text-sm">Beautiful success flows</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  <span className="text-sm">Real-time feature access</span>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Implementation Architecture</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <h4 className="font-semibold mb-2">Constants Layer</h4>
                  <ul className="space-y-1 text-muted-foreground">
                    <li>• constants/tiers.ts</li>
                    <li>• constants/featureMatrix.ts</li>
                    <li>• Single source of truth</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Component Layer</h4>
                  <ul className="space-y-1 text-muted-foreground">
                    <li>• ConversionTierCard</li>
                    <li>• CanonicalPricingPage</li>
                    <li>• PaymentSuccess</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Backend Layer</h4>
                  <ul className="space-y-1 text-muted-foreground">
                    <li>• Supabase edge functions</li>
                    <li>• Stripe integration</li>
                    <li>• Secure checkout flow</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default TierSystemSummary;