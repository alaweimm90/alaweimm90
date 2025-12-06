import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Button } from "@/ui/atoms/Button";
import { Badge } from "@/ui/atoms/Badge";
import { Check, Star, Zap, Crown } from 'lucide-react';
import { useMediaQuery } from '@/hooks/useMediaQuery';
import { useMobileOptimization } from '@/hooks/useMobileOptimization';
import { cn } from '@/lib/utils';

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

interface ResponsiveTierDisplayProps {
  onSelectTier?: (tierId: string) => void;
  currentTier?: string;
  className?: string;
}

const tierData: TierData[] = [
  {
    id: 'core',
    name: 'Core Program',
    description: 'Essential training foundation',
    price: { monthly: 89, annual: 890 },
    icon: <Star className="h-5 w-5" />,
    gradient: 'from-blue-500 to-blue-600',
    buttonText: 'Choose Core Program',
    testId: 'tier-card-core',
    features: [
      { name: 'Personalized Training Plan', included: true },
      { name: 'Nutrition Plan', included: true },
      { name: 'Static Dashboard', included: true },
      { name: 'Q&A Access - Limited', included: true },
      { name: 'Response Time - 72hr', included: true },
      { name: 'Science-based Tips', included: true },
      { name: 'Supplements Guide', included: true },
      { name: 'Weekly Check-ins & Photos', included: false },
      { name: 'Form Review', included: false },
      { name: 'Wearable Sync', included: false },
      { name: 'Auto Grocery Lists', included: false },
      { name: 'Travel Workouts', included: false },
      { name: 'Biomarker Integration', included: false },
      { name: 'AI Fitness Assistant', included: false },
      { name: 'Peptides Protocols', included: false },
      { name: 'PEDs Protocols', included: false },
      { name: 'Nootropics & Productivity', included: false },
      { name: 'Bioregulators', included: false },
      { name: 'Private Telegram Community', included: false },
      { name: 'In-person Training', included: false }
    ]
  },
  {
    id: 'adaptive',
    name: 'Adaptive Engine',
    description: 'Interactive coaching with tracking',
    price: { monthly: 149, annual: 1490 },
    popular: true,
    icon: <Zap className="h-5 w-5" />,
    gradient: 'from-orange-500 to-orange-600',
    buttonText: 'Choose Adaptive Engine',
    testId: 'tier-card-adaptive',
    features: [
      { name: 'Personalized Training Plan', included: true },
      { name: 'Nutrition Plan', included: true },
      { name: 'Interactive Dashboard', included: true, highlight: true },
      { name: 'Q&A Access - Standard', included: true },
      { name: 'Response Time - 48hr', included: true },
      { name: 'Science-based Tips', included: true },
      { name: 'Supplements Guide', included: true },
      { name: 'Weekly Check-ins & Photos', included: true, highlight: true },
      { name: 'Form Review', included: true, highlight: true },
      { name: 'Wearable Sync', included: true, highlight: true },
      { name: 'Auto Grocery Lists', included: true, highlight: true },
      { name: 'Travel Workouts', included: true, highlight: true },
      { name: 'Biomarker Integration', included: true, highlight: true },
      { name: 'AI Fitness Assistant', included: false },
      { name: 'Peptides Protocols', included: true, highlight: true },
      { name: 'PEDs Protocols', included: false },
      { name: 'Nootropics & Productivity', included: false },
      { name: 'Bioregulators', included: true, highlight: true },
      { name: 'Private Telegram Community', included: true, highlight: true },
      { name: 'In-person Training', included: false }
    ]
  },
  {
    id: 'performance',
    name: 'Prime Suite',
    description: 'Advanced AI and enhancement protocols',
    price: { monthly: 229, annual: 2290 },
    icon: <Zap className="h-5 w-5" />,
    gradient: 'from-purple-500 to-purple-600',
    buttonText: 'Choose Prime Suite',
    testId: 'tier-card-performance',
    features: [
      { name: 'Personalized Training Plan', included: true },
      { name: 'Nutrition Plan', included: true },
      { name: 'Interactive Dashboard', included: true },
      { name: 'Q&A Access - Priority', included: true },
      { name: 'Response Time - 24hr', included: true },
      { name: 'Science-based Tips', included: true },
      { name: 'Supplements Guide', included: true },
      { name: 'Weekly Check-ins & Photos', included: true },
      { name: 'Form Review', included: true },
      { name: 'Wearable Sync', included: true },
      { name: 'Auto Grocery Lists', included: true },
      { name: 'Travel Workouts', included: true },
      { name: 'Biomarker Integration', included: true },
      { name: 'AI Fitness Assistant', included: true, highlight: true },
      { name: 'Peptides Protocols', included: true },
      { name: 'PEDs Protocols', included: true, highlight: true },
      { name: 'Nootropics & Productivity', included: true, highlight: true },
      { name: 'Bioregulators', included: true },
      { name: 'Private Telegram Community', included: true },
      { name: 'In-person Training - 50% Off', included: true, highlight: true }
    ]
  },
  {
    id: 'longevity',
    name: 'Elite Concierge',
    description: 'Premium concierge with in-person training',
    price: { monthly: 349, annual: 3490 },
    premium: true,
    icon: <Crown className="h-5 w-5" />,
    gradient: 'from-yellow-500 to-yellow-600',
    buttonText: 'Choose Elite Concierge',
    testId: 'tier-card-longevity',
    features: [
      { name: 'Personalized Training Plan', included: true },
      { name: 'Nutrition Plan', included: true },
      { name: 'Interactive Dashboard', included: true },
      { name: 'Q&A Access - Unlimited', included: true, highlight: true },
      { name: 'Response Time - 12hr', included: true, highlight: true },
      { name: 'Science-based Tips', included: true },
      { name: 'Supplements Guide', included: true },
      { name: 'Weekly Check-ins & Photos', included: true },
      { name: 'Form Review', included: true },
      { name: 'Wearable Sync', included: true },
      { name: 'Auto Grocery Lists', included: true },
      { name: 'Travel Workouts', included: true },
      { name: 'Biomarker Integration', included: true },
      { name: 'AI Fitness Assistant', included: true },
      { name: 'Peptides Protocols', included: true },
      { name: 'PEDs Protocols', included: true },
      { name: 'Nootropics & Productivity', included: true },
      { name: 'Bioregulators', included: true },
      { name: 'Private Telegram Community', included: true },
      { name: 'In-person Training - Included (2×/week)', included: true, highlight: true }
    ]
  }
];

export function ResponsiveTierDisplay({ 
  onSelectTier, 
  currentTier,
  className 
}: ResponsiveTierDisplayProps) {
  const isMobile = useMediaQuery('(max-width: 768px)');
  const isTablet = useMediaQuery('(min-width: 769px) and (max-width: 1023px)');
  const mobileOptimization = useMobileOptimization();

  // Responsive grid classes
  const getGridClasses = () => {
    if (isMobile) {
      return 'grid grid-cols-1 gap-6';
    } else if (isTablet) {
      return 'grid grid-cols-2 gap-6';
    } else {
      return 'grid grid-cols-4 gap-6';
    }
  };

  // Responsive card classes
  const getCardClasses = (tier: TierData) => {
    const baseClasses = [
      'relative',
      'transition-all duration-300',
      'hover:scale-105',
      'group',
      'tier-card' // For testing selector
    ];

    if (tier.popular) {
      baseClasses.push('ring-2 ring-primary ring-offset-2');
    }

    if (tier.premium) {
      baseClasses.push('ring-2 ring-accent ring-offset-2');
    }

    if (currentTier === tier.id) {
      baseClasses.push('ring-2 ring-success ring-offset-2');
    }

    // Mobile-specific classes
    if (isMobile) {
      baseClasses.push('w-full');
    }

    return cn(baseClasses, className);
  };

  // Responsive button classes
  const getButtonClasses = (tier: TierData) => {
    const baseClasses = [
      'w-full',
      'transition-all duration-200'
    ];

    if (mobileOptimization.capabilities.isTouch) {
      baseClasses.push('min-h-[44px]'); // Touch target minimum
    }

    if (tier.popular) {
      baseClasses.push('bg-primary hover:bg-primary/90');
    } else if (tier.premium) {
      baseClasses.push('bg-accent hover:bg-accent/90');
    }

    return cn(baseClasses);
  };

  return (
    <div className={cn('w-full', className)}>
      {/* Popular tier indicator for mobile */}
      {isMobile && (
        <div className="mb-6 text-center">
          <Badge variant="default" className="animate-pulse">
            Most Popular: Adaptive Tier
          </Badge>
        </div>
      )}

      <div className={getGridClasses()}>
        {tierData.map((tier) => (
          <Card 
            key={tier.id} 
            className={getCardClasses(tier)}
            data-testid={tier.testId}
          >
            {/* Popular badge */}
            {tier.popular && !isMobile && (
              <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 z-10">
                <Badge variant="default" className="animate-pulse">
                  Most Popular
                </Badge>
              </div>
            )}

            {/* Premium badge */}
            {tier.premium && (
              <div className="absolute -top-3 right-4 z-10">
                <Badge variant="secondary" className="bg-accent text-accent-foreground">
                  Elite
                </Badge>
              </div>
            )}

            <CardHeader className="pb-4">
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className={cn(
                    'p-2 rounded-lg bg-gradient-to-r',
                    tier.gradient,
                    'text-white'
                  )}>
                    {tier.icon}
                  </div>
                  <div>
                    <CardTitle className="text-xl">{tier.name}</CardTitle>
                    <p className="text-sm text-muted-foreground">
                      {tier.description}
                    </p>
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="flex items-baseline gap-1">
                    <span className="text-3xl font-bold">
                      ${tier.price.monthly}
                    </span>
                    <span className="text-muted-foreground">/month</span>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    or ${tier.price.annual}/year (save ${(tier.price.monthly * 12 - tier.price.annual)})
                  </div>
                </div>
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              <div className="space-y-2">
                {tier.features.map((feature, index) => (
                  <div 
                    key={index} 
                    className={cn(
                      'flex items-center gap-2 text-sm',
                      feature.included ? 'text-foreground' : 'text-muted-foreground'
                    )}
                  >
                    <Check 
                      className={cn(
                        'h-4 w-4 flex-shrink-0',
                        feature.included 
                          ? feature.highlight 
                            ? 'text-primary' 
                            : 'text-success'
                          : 'text-muted-foreground opacity-30'
                      )} 
                    />
                    <span className={cn(
                      feature.highlight && feature.included ? 'font-medium text-primary' : '',
                      !feature.included ? 'line-through' : ''
                    )}>
                      {feature.name}
                    </span>
                  </div>
                ))}
              </div>

              <Button 
                className={getButtonClasses(tier)}
                onClick={() => onSelectTier?.(tier.id)}
                variant={tier.popular ? "default" : tier.premium ? "secondary" : "outline"}
                size={isMobile ? "lg" : "default"}
              >
                {currentTier === tier.id ? 'Current Plan' : tier.buttonText}
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Mobile-specific footer */}
      {isMobile && (
        <div className="mt-8 p-4 bg-muted/50 rounded-lg">
          <div className="text-center space-y-2">
            <p className="text-sm font-medium">Need help choosing?</p>
            <p className="text-xs text-muted-foreground">
              Most members start with Adaptive and upgrade as needed
            </p>
          </div>
        </div>
      )}
    </div>
  );
}