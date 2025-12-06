import React, { memo } from 'react';
import { Button } from "@/ui/atoms/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Badge } from "@/ui/atoms/Badge";
import { RepzLogo } from '@/ui/organisms/RepzLogo';

// Shared training options data
export const TRAINING_OPTIONS = [
  {
    id: 'city_sports',
    title: 'City Sports Club',
    price: 85,
    duration: '60 min',
    description: 'Premium facility with full equipment access',
    features: [
      'State-of-the-art equipment',
      'Professional training environment',
      'Locker room and shower facilities',
      'Free parking available',
      'Nutritional guidance included'
    ],
    location: 'Downtown Sports Complex',
    availability: 'Mon-Fri: 6 AM - 10 PM, Weekends: 8 AM - 8 PM'
  },
  {
    id: 'other_gyms',
    title: 'Other Gyms',
    price: 100,
    duration: '60 min',
    description: 'Train at your preferred local gym',
    features: [
      'Flexible location options',
      'Work with your existing gym membership',
      'Equipment familiarity advantage',
      'Convenient scheduling',
      'Personalized workout adaptations'
    ],
    location: 'Your preferred gym location',
    availability: 'Flexible scheduling based on gym hours'
  },
  {
    id: 'home_gym',
    title: 'Home Gym',
    price: 100,
    duration: '60 min',
    description: 'Personal training in your home environment',
    features: [
      'Ultimate privacy and comfort',
      'No travel time required',
      'Equipment recommendations provided',
      'Family members can join',
      'Customized to your space'
    ],
    location: 'Your home',
    availability: 'Flexible scheduling 7 days a week'
  }
];

// Base policies and guarantees
export const POLICIES = {
  guarantee: "30-day satisfaction guarantee",
  cancellation: "Cancel anytime with 24-hour notice",
  rescheduling: "Free rescheduling up to 2 hours before session",
  equipment: "All necessary equipment provided or recommended"
};

export interface TrainingVariationTheme {
  primary: string;
  secondary: string;
  accent: string;
  background: string;
  cardBackground: string;
  textPrimary: string;
  textSecondary: string;
  gradients?: {
    primary?: string;
    card?: string;
  };
}

interface TrainingVariationBaseProps {
  theme: TrainingVariationTheme;
  title: string;
  subtitle: string;
  selectedTier: string;
  onTierSelect: (tier: string) => void;
  customContent?: React.ReactNode;
  customPricingCard?: (option: typeof TRAINING_OPTIONS[0], isSelected: boolean) => React.ReactNode;
  className?: string;
}

const TrainingVariationBase: React.FC<TrainingVariationBaseProps> = ({
  theme,
  title,
  subtitle,
  selectedTier,
  onTierSelect,
  customContent,
  customPricingCard,
  className = ""
}) => {
  const defaultPricingCard = (option: typeof TRAINING_OPTIONS[0], isSelected: boolean) => (
    <Card 
      key={option.id}
      className={`relative cursor-pointer transition-all duration-300 hover:shadow-lg border-2 ${
        isSelected ? 'shadow-xl' : 'hover:shadow-md'
      } ${className}`}
      style={{
        borderColor: isSelected ? theme.primary : 'transparent',
        backgroundColor: isSelected ? theme.cardBackground : theme.background
      }}
      onClick={() => onTierSelect(option.id)}
    >
      <CardHeader className="text-center pb-4">
        <CardTitle className="text-2xl font-bold" style={{ color: theme.textPrimary }}>
          {option.title}
        </CardTitle>
        <div className="space-y-2">
          <div className="text-4xl font-bold" style={{ color: theme.primary }}>
            ${option.price}
            <span className="text-lg font-normal" style={{ color: theme.textSecondary }}>
              /{option.duration}
            </span>
          </div>
          <p style={{ color: theme.textSecondary }}>{option.description}</p>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-6">
        <div className="space-y-3">
          {option.features.map((feature, index) => (
            <div key={index} className="flex items-start gap-3">
              <div 
                className="w-5 h-5 rounded-full flex items-center justify-center mt-0.5"
                style={{ backgroundColor: theme.primary }}
              >
                <div className="w-2 h-2 bg-white rounded-full" />
              </div>
              <span style={{ color: theme.textPrimary }}>{feature}</span>
            </div>
          ))}
        </div>
        
        <div className="pt-4 border-t space-y-2" style={{ borderColor: theme.secondary }}>
          <div className="text-sm" style={{ color: theme.textSecondary }}>
            <strong>Location:</strong> {option.location}
          </div>
          <div className="text-sm" style={{ color: theme.textSecondary }}>
            <strong>Availability:</strong> {option.availability}
          </div>
        </div>
        
        <Button
          className={`w-full py-3 font-semibold transition-all duration-200 ${
            isSelected ? 'text-white' : ''
          }`}
          style={{
            backgroundColor: isSelected ? theme.primary : 'transparent',
            borderColor: theme.primary,
            color: isSelected ? 'white' : theme.primary
          }}
          variant={isSelected ? "default" : "outline"}
         onClick={() => console.log("TrainingVariationBase button clicked")}>
          {isSelected ? 'Selected' : 'Select This Option'}
        </Button>
      </CardContent>
    </Card>
  );

  return (
    <div 
      className="min-h-screen"
      style={{ 
        background: theme.gradients?.primary || theme.background 
      }}
    >
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <RepzLogo className="mx-auto mb-8" />
          <h1 
            className="text-5xl font-bold mb-4"
            style={{ color: theme.textPrimary }}
          >
            {title}
          </h1>
          <p 
            className="text-xl max-w-3xl mx-auto"
            style={{ color: theme.textSecondary }}
          >
            {subtitle}
          </p>
        </div>

        {/* Custom Content */}
        {customContent}

        {/* Training Options */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
          {TRAINING_OPTIONS.map((option) => {
            const isSelected = selectedTier === option.id;
            return customPricingCard 
              ? customPricingCard(option, isSelected)
              : defaultPricingCard(option, isSelected);
          })}
        </div>

        {/* Policies */}
        <Card className="max-w-4xl mx-auto" style={{ backgroundColor: theme.cardBackground }}>
          <CardHeader>
            <CardTitle style={{ color: theme.textPrimary }}>Our Commitment to You</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <Badge style={{ backgroundColor: theme.primary }}>✓</Badge>
                  <div>
                    <h4 className="font-semibold" style={{ color: theme.textPrimary }}>
                      Satisfaction Guarantee
                    </h4>
                    <p style={{ color: theme.textSecondary }}>{POLICIES.guarantee}</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Badge style={{ backgroundColor: theme.primary }}>✓</Badge>
                  <div>
                    <h4 className="font-semibold" style={{ color: theme.textPrimary }}>
                      Flexible Cancellation
                    </h4>
                    <p style={{ color: theme.textSecondary }}>{POLICIES.cancellation}</p>
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <Badge style={{ backgroundColor: theme.primary }}>✓</Badge>
                  <div>
                    <h4 className="font-semibold" style={{ color: theme.textPrimary }}>
                      Easy Rescheduling
                    </h4>
                    <p style={{ color: theme.textSecondary }}>{POLICIES.rescheduling}</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Badge style={{ backgroundColor: theme.primary }}>✓</Badge>
                  <div>
                    <h4 className="font-semibold" style={{ color: theme.textPrimary }}>
                      Equipment Provided
                    </h4>
                    <p style={{ color: theme.textSecondary }}>{POLICIES.equipment}</p>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default memo(TrainingVariationBase);