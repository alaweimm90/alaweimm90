// src/components/intake/steps/Step3Health.tsx
import React from 'react';
import { SelectField } from "@/components/ui/form-fields";
import { Input } from "@/ui/atoms/Input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/ui/atoms/Checkbox";
import { Label } from "@/ui/atoms/Label";
import { Heart, AlertTriangle, Pill } from 'lucide-react';
import { useFormValidation, stepValidationRules } from '@/hooks/useFormValidation';

interface Step3FormData {
  medicalConditions?: string;
  injuries?: string;
  medications?: string;
  activityLevel?: string;
  [key: string]: string | undefined;
}

interface Step3HealthProps {
  formData: Step3FormData;
  onFieldChange: (field: string, value: string) => void;
}

export const Step3Health: React.FC<Step3HealthProps> = ({
  formData,
  onFieldChange
}) => {
  const { validateFormData } = useFormValidation();
  const validation = validateFormData(formData, stepValidationRules.step3);

  const handleMedicalConditionChange = (condition: string, checked: boolean) => {
    const currentConditions = formData.medicalConditions || [];
    if (checked) {
      onFieldChange('medicalConditions', [...currentConditions, condition]);
    } else {
      onFieldChange('medicalConditions', currentConditions.filter((c: string) => c !== condition));
    }
  };

  return (
    <div className="step-container w-full h-full">
      <div className="max-w-4xl mx-auto p-6 md:p-8">
        {/* Header */}
        <div className="text-center space-y-4 mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 tracking-tight">
            Health & Medical Information
          </h1>
          <p className="text-base md:text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Help us understand your health background for safe and effective coaching
          </p>
        </div>

        {/* Form Container */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm space-y-8">
          {/* Current Health Status */}
          <div className="p-6 md:p-8 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <Heart className="h-5 w-5 text-red-500" />
              <h3 className="text-lg font-semibold text-gray-900">Current Health Status</h3>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SelectField
                id="activityLevel"
                label="Current Activity Level"
                value={formData.activityLevel || ''}
                onChange={(value) => onFieldChange('activityLevel', value)}
                placeholder="Select activity level"
                required
                options={[
                  { value: 'sedentary', label: 'Sedentary (little to no exercise)' },
                  { value: 'lightly-active', label: 'Lightly Active (1-3 days/week)' },
                  { value: 'moderately-active', label: 'Moderately Active (3-5 days/week)' },
                  { value: 'very-active', label: 'Very Active (6-7 days/week)' },
                  { value: 'extremely-active', label: 'Extremely Active (2x/day or intense training)' }
                ]}
              />
              {validation.errors.activityLevel && (
                <p className="text-sm text-red-500">{validation.errors.activityLevel}</p>
              )}

              <SelectField
                id="sleepHours"
                label="Average Sleep Hours per Night"
                value={formData.sleepHours || ''}
                onChange={(value) => onFieldChange('sleepHours', value)}
                placeholder="Select sleep hours"
                required
                options={[
                  { value: 'less-than-5', label: 'Less than 5 hours' },
                  { value: '5-6', label: '5-6 hours' },
                  { value: '6-7', label: '6-7 hours' },
                  { value: '7-8', label: '7-8 hours' },
                  { value: '8-9', label: '8-9 hours' },
                  { value: 'more-than-9', label: 'More than 9 hours' }
                ]}
              />
              {validation.errors.sleepHours && (
                <p className="text-sm text-red-500">{validation.errors.sleepHours}</p>
              )}
            </div>
          </div>

          {/* Medical Conditions */}
          <div className="p-6 md:p-8 border-t border-gray-100 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <AlertTriangle className="h-5 w-5 text-repz-orange" />
              <h3 className="text-lg font-semibold text-gray-900">Medical History</h3>
            </div>
            
            <div className="space-y-6">
              <div>
                <Label className="text-sm font-medium text-gray-700 mb-4 block">
                  Do you have any of the following conditions? (Check all that apply)
                </Label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[
                    'Heart disease', 'High blood pressure', 'Diabetes', 'Arthritis',
                    'Back/spine issues', 'Knee problems', 'Shoulder injuries', 'Asthma',
                    'Thyroid issues', 'Depression/anxiety', 'Previous surgeries', 'None of the above'
                  ].map((condition) => (
                    <div key={condition} className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-50 transition-colors">
                      <Checkbox
                        id={condition}
                        checked={(formData.medicalConditions || []).includes(condition)}
                        onCheckedChange={(checked) => handleMedicalConditionChange(condition, checked as boolean)}
                        className="h-5 w-5 data-[state=checked]:bg-repz-orange data-[state=checked]:border-repz-orange"
                      />
                      <Label htmlFor={condition} className="text-sm font-medium text-gray-700 cursor-pointer">
                        {condition}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-3">
                <Label htmlFor="medicationDetails" className="text-sm font-medium text-gray-700">
                  Please provide details about any medications or medical conditions
                </Label>
                <Textarea
                  id="medicationDetails"
                  placeholder="List any medications, supplements, or provide details about medical conditions..."
                  value={formData.medicationDetails || ''}
                  onChange={(e) => onFieldChange('medicationDetails', e.target.value)}
                  className="min-h-[100px] border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                />
              </div>
            </div>
          </div>

          {/* Injuries & Limitations */}
          <div className="p-6 md:p-8 border-t border-gray-100 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <Pill className="h-5 w-5 text-blue-500" />
              <h3 className="text-lg font-semibold text-gray-900">Injuries & Physical Limitations</h3>
            </div>
            
            <div className="space-y-6">
              <div className="space-y-3">
                <Label htmlFor="injuries" className="text-sm font-medium text-gray-700">
                  Current or past injuries that might affect training
                </Label>
                <Textarea
                  id="injuries"
                  placeholder="Describe any injuries, when they occurred, and current status..."
                  value={formData.injuries || ''}
                  onChange={(e) => onFieldChange('injuries', e.target.value)}
                  className="min-h-[100px] border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                />
              </div>

              <div className="space-y-3">
                <Label htmlFor="physicalLimitations" className="text-sm font-medium text-gray-700">
                  Any movements or exercises you cannot perform?
                </Label>
                <Textarea
                  id="physicalLimitations"
                  placeholder="List any exercises or movements to avoid..."
                  value={formData.physicalLimitations || ''}
                  onChange={(e) => onFieldChange('physicalLimitations', e.target.value)}
                  className="min-h-[100px] border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                />
              </div>
            </div>
          </div>

          {/* Emergency Contact */}
          <div className="p-6 md:p-8 border-t border-gray-100 space-y-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Emergency Contact</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-3">
                <Label htmlFor="emergencyContactName" className="text-sm font-medium text-gray-700">
                  Emergency Contact Name
                </Label>
                <Input
                  id="emergencyContactName"
                  type="text"
                  placeholder="Full name"
                  value={formData.emergencyContactName || ''}
                  onChange={(e) => onFieldChange('emergencyContactName', e.target.value)}
                  className="h-12 px-4 border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                />
              </div>

              <div className="space-y-3">
                <Label htmlFor="emergencyContactPhone" className="text-sm font-medium text-gray-700">
                  Emergency Contact Phone
                </Label>
                <Input
                  id="emergencyContactPhone"
                  type="tel"
                  placeholder="Phone number"
                  value={formData.emergencyContactPhone || ''}
                  onChange={(e) => onFieldChange('emergencyContactPhone', e.target.value)}
                  className="h-12 px-4 border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};