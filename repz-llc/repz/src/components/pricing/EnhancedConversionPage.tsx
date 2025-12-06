import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Check, Zap, Clock, MessageSquare, Users } from 'lucide-react';
import { BillingCycle } from '@/constants/tiers';
import { ComprehensiveBillingSelector } from '@/components/ui/comprehensive-billing-selector';
import { cn } from '@/lib/utils';

interface EnhancedConversionPageProps {
  onTierSelect?: (tier: string, billing: BillingCycle) => void;
}

const TIER_PSYCHOLOGY = {
  core: {
    name: 'Core Foundation',
    monthlyPrice: 89,
    description: 'Perfect for beginners starting their fitness journey',
    badge: 'FOUNDATION',
    badgeColor: 'bg-blue-500',
    features: [
      'Personalized training program',
      'Nutrition guidance',
      'Basic progress tracking',
      '72-hour coach response',
      'Monthly check-ins'
    ],
    psychology: {
      anchor: 'Most affordable entry point',
      social: '40% of clients start here',
      scarcity: null
    }
  },
  adaptive: {
    name: 'Adaptive Growth',
    monthlyPrice: 149,
    description: 'Most popular for serious fitness enthusiasts',
    badge: 'MOST POPULAR',
    badgeColor: 'bg-orange-500',
    features: [
      'Everything in Core +',
      'Weekly check-ins & adjustments',
      'Form analysis videos',
      'Wearable device integration',
      '48-hour coach response',
      'Supplement protocols'
    ],
    psychology: {
      anchor: 'Best value for results',
      social: '60% choose this tier',
      scarcity: null
    }
  },
  performance: {
    name: 'Performance Suite',
    monthlyPrice: 229,
    description: 'For athletes and performance-focused individuals',
    badge: 'PERFORMANCE',
    badgeColor: 'bg-purple-500',
    features: [
      'Everything in Adaptive +',
      'AI fitness assistant',
      'Biomarker integration',
      'Advanced analytics',
      '24-hour coach response',
      'Peptide protocols'
    ],
    psychology: {
      anchor: 'Professional athlete choice',
      social: 'Preferred by competitive athletes',
      scarcity: null
    }
  },
  longevity: {
    name: 'Longevity Concierge',
    monthlyPrice: 349,
    description: 'Ultimate optimization and longevity protocols',
    badge: 'EXCLUSIVE',
    badgeColor: 'bg-yellow-500',
    features: [
      'Everything in Performance +',
      'In-person training (SF Bay Area)',
      'Bioregulator protocols',
      '12-hour coach response',
      'Concierge health optimization',
      'Advanced longevity protocols'
    ],
    psychology: {
      anchor: 'Executive & entrepreneur tier',
      social: 'Limited to 50 clients globally',
      scarcity: 'Only 12 spots remaining'
    }
  }
};

export const EnhancedConversionPage: React.FC<EnhancedConversionPageProps> = ({
  onTierSelect = () => {}
}) => {
  const [selectedBilling, setSelectedBilling] = React.useState<BillingCycle>('annual');

  const handleTierSelection = (tierKey: string) => {
    onTierSelect(tierKey, selectedBilling);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black">
      {/* Hero Section */}
      <section className="py-16 px-4">
        <div className="container mx-auto text-center">
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
            Choose Your 
            <span className="block text-orange-400">Transformation Path</span>
          </h1>
          
          <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-8">
            Join thousands of clients who've transformed their fitness with our science-based coaching methodology
          </p>

          <div className="flex items-center justify-center gap-6 text-sm text-gray-400 mb-12">
            <div className="flex items-center gap-2">
              <Users className="h-4 w-4 text-orange-400" />
              <span>5,000+ Successful Transformations</span>
            </div>
            <span>•</span>
            <div className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4 text-orange-400" />
              <span>Expert Coaching</span>
            </div>
            <span>•</span>
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-orange-400" />
              <span>Proven Results</span>
            </div>
          </div>
        </div>
      </section>

      {/* Billing Selector */}
      <section className="py-8 px-4">
        <div className="container mx-auto max-w-4xl">
          <ComprehensiveBillingSelector
            selectedCycle={selectedBilling}
            onCycleChange={setSelectedBilling}
            monthlyPrice={149} // Use adaptive as base for calculation
            showAnchoringDefault={true}
          />
        </div>
      </section>

      {/* Tier Cards */}
      <section className="py-12 px-4">
        <div className="container mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {Object.entries(TIER_PSYCHOLOGY).map(([key, tier], index) => {
              const savings = selectedBilling === 'annual' ? Math.round(tier.monthlyPrice * 12 * 0.2) : 0;
              const effectivePrice = selectedBilling === 'annual' 
                ? tier.monthlyPrice * 0.8 
                : tier.monthlyPrice;

              return (
                <Card
                  key={key}
                  className={cn(
                    "relative overflow-hidden transition-all duration-300 hover:scale-105",
                    "bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-sm",
                    "border-2",
                    key === 'adaptive' && "border-orange-500/50 ring-2 ring-orange-200/20",
                    key !== 'adaptive' && "border-white/10 hover:border-white/20"
                  )}
                >
                  {/* Badge */}
                  <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 z-10">
                    <Badge className={cn(tier.badgeColor, "text-white px-3 py-1 text-xs font-bold")}>
                      {tier.badge}
                    </Badge>
                  </div>

                  {/* Scarcity Indicator */}
                  {tier.psychology.scarcity && (
                    <div className="absolute top-4 right-4 z-10">
                      <div className="bg-red-500/20 border border-red-500/30 rounded-full px-2 py-1">
                        <span className="text-xs text-red-400 font-medium">
                          {tier.psychology.scarcity}
                        </span>
                      </div>
                    </div>
                  )}

                  <CardHeader className="pb-4 pt-8">
                    <CardTitle className="text-xl font-bold text-white text-center">
                      {tier.name}
                    </CardTitle>
                    <CardDescription className="text-center text-gray-300">
                      {tier.description}
                    </CardDescription>

                    {/* Pricing */}
                    <div className="text-center mt-4">
                      <div className="flex items-center justify-center gap-2">
                        <span className="text-3xl font-bold text-white">
                          ${Math.round(effectivePrice)}
                        </span>
                        <span className="text-sm text-gray-400">/month</span>
                      </div>
                      
                      {selectedBilling === 'annual' && savings > 0 && (
                        <div className="text-green-400 text-sm mt-1">
                          Save ${savings}/year
                        </div>
                      )}
                      
                      {selectedBilling !== 'monthly' && (
                        <div className="text-xs text-gray-500 mt-1">
                          Billed {selectedBilling}
                        </div>
                      )}
                    </div>

                    {/* Social Proof */}
                    <div className="text-center mt-3">
                      <span className="text-xs text-gray-400 bg-white/5 px-2 py-1 rounded-full">
                        {tier.psychology.social}
                      </span>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    {/* Features */}
                    <div className="space-y-2">
                      {tier.features.map((feature, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm">
                          <Check className="h-4 w-4 text-green-400 flex-shrink-0" />
                          <span className="text-gray-300">{feature}</span>
                        </div>
                      ))}
                    </div>

                    {/* Response Time Highlight */}
                    <div className="flex items-center gap-2 p-3 bg-white/5 rounded-lg">
                      <Clock className="h-4 w-4 text-orange-400" />
                      <span className="text-sm text-gray-300">
                        {tier.features.find(f => f.includes('response'))?.split('response')[0]}response guarantee
                      </span>
                    </div>

                    {/* CTA Button */}
                    <Button
                      onClick={() => handleTierSelection(key)}
                      className={cn(
                        "w-full mt-6 font-semibold transition-all duration-200",
                        key === 'adaptive' 
                          ? "bg-orange-500 hover:bg-orange-600 text-white" 
                          : "bg-white/10 hover:bg-white/20 text-white border border-white/20"
                      )}
                    >
                      {key === 'adaptive' ? 'Most Popular Choice' : `Choose ${tier.name}`}
                    </Button>

                    {/* Psychology Element */}
                    <div className="text-center text-xs text-gray-500 mt-2">
                      {tier.psychology.anchor}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* Social Proof & Benefits */}
      <section className="py-16 px-4">
        <div className="container mx-auto text-center">
          <h2 className="text-3xl font-bold text-white mb-8">
            Trusted by Professionals Worldwide
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-6">
              <div className="text-4xl font-bold text-orange-400 mb-2">5,000+</div>
              <div className="text-white font-semibold mb-1">Clients Transformed</div>
              <div className="text-gray-400 text-sm">Across all fitness levels</div>
            </div>
            
            <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-6">
              <div className="text-4xl font-bold text-orange-400 mb-2">98%</div>
              <div className="text-white font-semibold mb-1">Success Rate</div>
              <div className="text-gray-400 text-sm">Clients reaching goals</div>
            </div>
            
            <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-6">
              <div className="text-4xl font-bold text-orange-400 mb-2">24/7</div>
              <div className="text-white font-semibold mb-1">Expert Support</div>
              <div className="text-gray-400 text-sm">When you need it most</div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};