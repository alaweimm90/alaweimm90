// src/components/intake/steps/Step4Training.tsx
import React from 'react';
import { TextAreaField } from "@/components/ui/form-fields";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/ui/molecules/Select";
import { Checkbox } from "@/ui/atoms/Checkbox";
import { Label } from "@/ui/atoms/Label";
import { Dumbbell, Target, Calendar } from 'lucide-react';
import { useFormValidation, stepValidationRules } from '@/hooks/useFormValidation';

interface Step4FormData {
  trainingExperience?: string;
  fitnessGoals?: string;
  workoutFrequency?: string;
  preferredTrainingStyle?: string;
  [key: string]: string | undefined;
}

interface Step4TrainingProps {
  formData: Step4FormData;
  onFieldChange: (field: string, value: string) => void;
}

export const Step4Training: React.FC<Step4TrainingProps> = ({
  formData,
  onFieldChange
}) => {
  const { validateFormData } = useFormValidation();
  const validation = validateFormData(formData, stepValidationRules.step4);

  const handleGoalChange = (goal: string, checked: boolean) => {
    const currentGoals = formData.primaryGoals || [];
    if (checked) {
      onFieldChange('primaryGoals', [...currentGoals, goal]);
    } else {
      onFieldChange('primaryGoals', currentGoals.filter((g: string) => g !== goal));
    }
  };

  return (
    <div className="step-container w-full h-full">
      <div className="max-w-4xl mx-auto p-6 md:p-8">
        {/* Header */}
        <div className="text-center space-y-4 mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 tracking-tight">
            Training Experience & Goals
          </h1>
          <p className="text-base md:text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Tell us about your fitness background and what you want to achieve
          </p>
        </div>

        {/* Form Container */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm space-y-8">
          {/* Primary Goals */}
          <div className="p-6 md:p-8 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <Target className="h-5 w-5 text-green-500" />
              <h3 className="text-lg font-semibold text-gray-900">Primary Goals</h3>
            </div>
            
            <div className="space-y-6">
              <div>
                <Label className="text-sm font-medium text-gray-700 mb-4 block">
                  What are your main fitness goals? (Select all that apply)
                </Label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[
                    'Weight loss', 'Muscle gain', 'Strength building', 'Endurance improvement',
                    'Athletic performance', 'General fitness', 'Injury rehabilitation', 'Stress relief',
                    'Better sleep', 'Increased energy', 'Body composition', 'Flexibility/mobility'
                  ].map((goal) => (
                    <div key={goal} className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-50 transition-colors">
                      <Checkbox
                        id={goal}
                        checked={(formData.primaryGoals || []).includes(goal)}
                        onCheckedChange={(checked) => handleGoalChange(goal, checked as boolean)}
                        className="h-5 w-5 data-[state=checked]:bg-repz-orange data-[state=checked]:border-repz-orange"
                      />
                      <Label htmlFor={goal} className="text-sm font-medium text-gray-700 cursor-pointer">
                        {goal}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              <TextAreaField
                id="specificGoals"
                label="Describe your specific goals in detail"
                placeholder="What exactly do you want to achieve? Be as specific as possible..."
                value={formData.specificGoals || ''}
                onChange={(value) => onFieldChange('specificGoals', value)}
                required
                rows={5}
                description='Examples: "Lose 20 pounds in 6 months", "Increase bench press by 50 lbs", "Run a 5K without stopping"'
              />
              {validation.errors.specificGoals && (
                <p className="text-sm text-red-500">{validation.errors.specificGoals}</p>
              )}
            </div>
          </div>

          {/* Training Experience */}
          <div className="p-6 md:p-8 border-t border-gray-100 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <Dumbbell className="h-5 w-5 text-blue-500" />
              <h3 className="text-lg font-semibold text-gray-900">Training Experience</h3>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-3">
                <Label htmlFor="fitnessLevel" className="text-sm font-medium text-gray-700">
                  Current Fitness Level *
                </Label>
                <Select value={formData.fitnessLevel || ''} onValueChange={(value) => onFieldChange('fitnessLevel', value)}>
                  <SelectTrigger className="h-12 border-gray-300 focus:border-repz-orange focus:ring-repz-orange">
                    <SelectValue placeholder="Select fitness level" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="beginner">Beginner (0-1 years of training)</SelectItem>
                    <SelectItem value="intermediate">Intermediate (1-3 years of training)</SelectItem>
                    <SelectItem value="advanced">Advanced (3+ years of training)</SelectItem>
                    <SelectItem value="expert">Expert/Competitive athlete</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <Label htmlFor="trainingFrequency" className="text-sm font-medium text-gray-700">
                  How often do you currently train? *
                </Label>
                <Select value={formData.trainingFrequency || ''} onValueChange={(value) => onFieldChange('trainingFrequency', value)}>
                  <SelectTrigger className="h-12 border-gray-300 focus:border-repz-orange focus:ring-repz-orange">
                    <SelectValue placeholder="Select frequency" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">Not currently training</SelectItem>
                    <SelectItem value="1-2">1-2 times per week</SelectItem>
                    <SelectItem value="3-4">3-4 times per week</SelectItem>
                    <SelectItem value="5-6">5-6 times per week</SelectItem>
                    <SelectItem value="daily">Daily</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-3">
              <Label htmlFor="trainingHistory" className="text-sm font-medium text-gray-700">
                Training History & Experience
              </Label>
              <Textarea
                id="trainingHistory"
                placeholder="Describe your training background, what types of exercise you've done, any sports experience..."
                value={formData.trainingHistory || ''}
                onChange={(e) => onFieldChange('trainingHistory', e.target.value)}
                className="min-h-[100px] border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
              />
            </div>
          </div>

          {/* Schedule & Preferences */}
          <div className="p-6 md:p-8 border-t border-gray-100 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <Calendar className="h-5 w-5 text-purple-500" />
              <h3 className="text-lg font-semibold text-gray-900">Schedule & Preferences</h3>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-3">
                <Label htmlFor="availableTime" className="text-sm font-medium text-gray-700">
                  How much time can you dedicate to training? *
                </Label>
                <Select value={formData.availableTime || ''} onValueChange={(value) => onFieldChange('availableTime', value)}>
                  <SelectTrigger className="h-12 border-gray-300 focus:border-repz-orange focus:ring-repz-orange">
                    <SelectValue placeholder="Select time commitment" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="30-45-min">30-45 minutes per session</SelectItem>
                    <SelectItem value="45-60-min">45-60 minutes per session</SelectItem>
                    <SelectItem value="60-90-min">60-90 minutes per session</SelectItem>
                    <SelectItem value="90-plus-min">90+ minutes per session</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <Label htmlFor="preferredSchedule" className="text-sm font-medium text-gray-700">
                  Preferred Training Days
                </Label>
                <Select value={formData.preferredSchedule || ''} onValueChange={(value) => onFieldChange('preferredSchedule', value)}>
                  <SelectTrigger className="h-12 border-gray-300 focus:border-repz-orange focus:ring-repz-orange">
                    <SelectValue placeholder="Select schedule" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="weekdays">Weekdays only</SelectItem>
                    <SelectItem value="weekends">Weekends only</SelectItem>
                    <SelectItem value="mixed">Mix of weekdays and weekends</SelectItem>
                    <SelectItem value="flexible">Flexible schedule</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-3">
              <Label htmlFor="trainingPreferences" className="text-sm font-medium text-gray-700">
                Training Preferences & Equipment Access
              </Label>
              <Textarea
                id="trainingPreferences"
                placeholder="Describe your training preferences, gym access, home equipment, preferred exercise types..."
                value={formData.trainingPreferences || ''}
                onChange={(e) => onFieldChange('trainingPreferences', e.target.value)}
                className="min-h-[100px] border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
              />
              <p className="text-sm text-gray-500">
                Include details about gym membership, home equipment, exercise preferences, etc.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};