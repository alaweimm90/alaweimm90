import React from 'react';
import { TextAreaField, SelectField } from '@/components/ui/form-fields';

interface TierFormData {
  coreGoals?: string;
  coreExperience?: string;
  adaptiveGoals?: string;
  performanceGoals?: string;
  longevityGoals?: string;
  [key: string]: string | undefined;
}

interface TierQuestionsSectionProps {
  tierId: string;
  formData: TierFormData;
  onFieldChange: (field: string, value: string) => void;
}

export const TierQuestionsSection: React.FC<TierQuestionsSectionProps> = ({
  tierId,
  formData,
  onFieldChange
}) => {
  const renderQuestions = () => {
    switch (tierId) {
      case 'core':
        return (
          <div className="space-y-6">
            <TextAreaField
              id="coreGoals"
              label="What are your main priorities for this coaching program?"
              placeholder="Focus on building healthy habits, basic fitness foundation, weight management, etc."
              value={formData.coreGoals || ''}
              onChange={(value) => onFieldChange('coreGoals', value)}
              required
              rows={4}
            />
            
            <TextAreaField
              id="challengingAspects"
              label="What aspects of fitness/nutrition do you find most challenging?"
              placeholder="Consistency, motivation, meal planning, understanding proper form, etc."
              value={formData.challengingAspects || ''}
              onChange={(value) => onFieldChange('challengingAspects', value)}
              rows={3}
            />
          </div>
        );

      case 'adaptive':
        return (
          <div className="space-y-6">
            <TextAreaField
              id="performanceGoals"
              label="What specific performance goals do you want to achieve?"
              placeholder="Increase strength by X%, improve endurance, compete in events, etc."
              value={formData.performanceGoals || ''}
              onChange={(value) => onFieldChange('performanceGoals', value)}
              required
              rows={4}
            />
            
            <SelectField
              id="trackingPreference"
              label="How do you prefer to track progress?"
              value={formData.trackingPreference || ''}
              onChange={(value) => onFieldChange('trackingPreference', value)}
              placeholder="Select tracking method"
              required
              options={[
                { value: 'photos-measurements', label: 'Photos and body measurements' },
                { value: 'performance-metrics', label: 'Performance metrics and PRs' },
                { value: 'both', label: 'Both visual and performance tracking' },
                { value: 'simple', label: 'Simple progress check-ins' }
              ]}
            />
          </div>
        );

      case 'performance':
        return (
          <div className="space-y-6">
            <TextAreaField
              id="performanceAdvancedGoals"
              label="What advanced goals require precision coaching?"
              placeholder="Body composition changes, competition prep, advanced strength goals, etc."
              value={formData.performanceAdvancedGoals || ''}
              onChange={(value) => onFieldChange('performanceAdvancedGoals', value)}
              required
              rows={4}
            />
            
            <SelectField
              id="dataTracking"
              label="How comfortable are you with detailed data tracking?"
              value={formData.dataTracking || ''}
              onChange={(value) => onFieldChange('dataTracking', value)}
              placeholder="Select comfort level"
              required
              options={[
                { value: 'love-data', label: 'Love detailed tracking and analytics' },
                { value: 'comfortable', label: 'Comfortable with regular tracking' },
                { value: 'moderate', label: 'Moderate tracking is fine' },
                { value: 'minimal', label: 'Prefer minimal tracking' }
              ]}
            />

            <SelectField
              id="supplementInterest"
              label="Interest in advanced supplementation protocols?"
              value={formData.supplementInterest || ''}
              onChange={(value) => onFieldChange('supplementInterest', value)}
              placeholder="Select interest level"
              options={[
                { value: 'very-interested', label: 'Very interested in optimization' },
                { value: 'somewhat', label: 'Somewhat interested' },
                { value: 'basic-only', label: 'Basic supplements only' },
                { value: 'none', label: 'Prefer no supplements' }
              ]}
            />
          </div>
        );

      case 'longevity':
        return (
          <div className="space-y-6">
            <TextAreaField
              id="longevityGoals"
              label="What are your longevity and health span goals?"
              placeholder="Aging gracefully, disease prevention, cognitive health, mobility preservation, etc."
              value={formData.longevityGoals || ''}
              onChange={(value) => onFieldChange('longevityGoals', value)}
              required
              rows={4}
            />
            
            <SelectField
              id="healthOptimization"
              label="Areas of health optimization you're most interested in"
              value={formData.healthOptimization || ''}
              onChange={(value) => onFieldChange('healthOptimization', value)}
              placeholder="Select primary focus"
              required
              options={[
                { value: 'metabolic', label: 'Metabolic health and insulin sensitivity' },
                { value: 'cardiovascular', label: 'Cardiovascular health and VO2 max' },
                { value: 'cognitive', label: 'Cognitive function and brain health' },
                { value: 'hormonal', label: 'Hormonal optimization' },
                { value: 'all', label: 'Comprehensive optimization' }
              ]}
            />

            <SelectField
              id="advancedProtocols"
              label="Interest in advanced longevity protocols?"
              value={formData.advancedProtocols || ''}
              onChange={(value) => onFieldChange('advancedProtocols', value)}
              placeholder="Select interest level"
              options={[
                { value: 'cutting-edge', label: 'Very interested in cutting-edge protocols' },
                { value: 'evidence-based', label: 'Evidence-based approaches only' },
                { value: 'conservative', label: 'Conservative, proven methods' },
                { value: 'basic', label: 'Basic longevity principles' }
              ]}
            />

            <SelectField
              id="communicationFrequency"
              label="Preferred frequency for check-ins and adjustments"
              value={formData.communicationFrequency || ''}
              onChange={(value) => onFieldChange('communicationFrequency', value)}
              placeholder="Select frequency"
              required
              options={[
                { value: 'weekly', label: 'Weekly detailed check-ins' },
                { value: 'bi-weekly', label: 'Bi-weekly comprehensive reviews' },
                { value: 'monthly', label: 'Monthly deep-dive sessions' },
                { value: 'as-needed', label: 'As-needed basis' }
              ]}
            />
          </div>
        );

      default:
        return null;
    }
  };

  return renderQuestions();
};