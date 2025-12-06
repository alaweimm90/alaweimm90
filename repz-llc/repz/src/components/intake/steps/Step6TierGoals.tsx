// src/components/intake/steps/Step6TierGoals.tsx
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Badge } from "@/ui/atoms/Badge";
import { getTierConfig } from '@/constants/tiers';
import { UnifiedTierCard } from '@/components/ui/unified-tier-card';
import { TierQuestionsSection } from '../TierQuestionsSection';
import { SelectField, TextAreaField } from '@/components/ui/form-fields';

interface Step6FormData {
  selectedTier?: string;
  primaryGoals?: string;
  coachingPreferences?: string;
  [key: string]: string | undefined;
}

interface Step6TierGoalsProps {
  formData: Step6FormData;
  onFieldChange: (field: string, value: string) => void;
  selectedTier: string;
}

export const Step6TierGoals: React.FC<Step6TierGoalsProps> = ({
  formData,
  onFieldChange,
  selectedTier
}) => {
  const tierConfig = getTierConfig(selectedTier);

  return (
    <div className="step-container w-full h-full">
      <div className="max-w-4xl mx-auto p-6 md:p-8">
        {/* Header */}
        <div className="text-center space-y-4 mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 tracking-tight">
            Tier-Specific Goals & Preferences
          </h1>
          <p className="text-base md:text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Let's customize your experience based on your selected coaching tier
          </p>
        </div>

        {/* Selected Tier Display */}
        <div className="mb-8">
          <UnifiedTierCard tier={selectedTier} variant="comparison" showFeatures={true} />
        </div>

        {/* Form Container */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm space-y-8">
          {/* Tier-Specific Questions */}
          <div className="p-6 md:p-8 space-y-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-6 border-b border-gray-100 pb-3">
              Customization for Your Tier
            </h3>
            <TierQuestionsSection 
              tierId={tierConfig.id}
              formData={formData}
              onFieldChange={onFieldChange}
            />
          </div>

          {/* General Preferences */}
          <div className="p-6 md:p-8 border-t border-gray-100 space-y-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-6 border-b border-gray-100 pb-3">
              Communication & Delivery Preferences
            </h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SelectField
                id="communicationStyle"
                label="Preferred communication style"
                value={formData.communicationStyle || ''}
                onChange={(value) => onFieldChange('communicationStyle', value)}
                placeholder="Select style"
                required
                options={[
                  { value: 'detailed', label: 'Detailed explanations and science' },
                  { value: 'practical', label: 'Practical and action-focused' },
                  { value: 'motivational', label: 'Motivational and encouraging' },
                  { value: 'direct', label: 'Direct and to-the-point' }
                ]}
              />

              <SelectField
                id="motivationStyle"
                label="What motivates you most?"
                value={formData.motivationStyle || ''}
                onChange={(value) => onFieldChange('motivationStyle', value)}
                placeholder="Select motivation"
                required
                options={[
                  { value: 'progress-tracking', label: 'Seeing measurable progress' },
                  { value: 'accountability', label: 'Regular accountability check-ins' },
                  { value: 'education', label: 'Learning and understanding the why' },
                  { value: 'community', label: 'Support and community feeling' }
                ]}
              />
            </div>

            <TextAreaField
              id="additionalNotes"
              label="Additional notes or special requests"
              placeholder="Anything else you'd like your coach to know about your goals, preferences, or situation..."
              value={formData.additionalNotes || ''}
              onChange={(value) => onFieldChange('additionalNotes', value)}
              description="Include any specific requirements, concerns, or questions you have"
              rows={4}
            />
          </div>
        </div>
      </div>
    </div>
  );
};