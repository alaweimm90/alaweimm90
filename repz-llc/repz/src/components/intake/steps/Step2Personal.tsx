// src/components/intake/steps/Step2Personal.tsx
import React from 'react';
import { SelectField } from "@/components/ui/form-fields";
import { Input } from "@/ui/atoms/Input";
import { Label } from "@/ui/atoms/Label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/ui/molecules/Select";
import { User, MapPin, Calendar } from 'lucide-react';
import { useFormValidation, stepValidationRules } from '@/hooks/useFormValidation';

interface Step2FormData {
  dateOfBirth?: string;
  gender?: string;
  location?: string;
  [key: string]: string | undefined;
}

interface Step2PersonalProps {
  formData: Step2FormData;
  onFieldChange: (field: string, value: string) => void;
}

export const Step2Personal: React.FC<Step2PersonalProps> = ({
  formData,
  onFieldChange
}) => {
  const { validateFormData } = useFormValidation();
  const validation = validateFormData(formData, stepValidationRules.step2);

  const calculateAge = (birthDate: string) => {
    if (!birthDate) return '';
    const today = new Date();
    const birth = new Date(birthDate);
    const age = today.getFullYear() - birth.getFullYear();
    const monthDiff = today.getMonth() - birth.getMonth();
    
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
      return (age - 1).toString();
    }
    return age.toString();
  };

  const toggleWeightUnit = () => {
    const currentUnit = formData.weightUnit || 'lbs';
    const newUnit = currentUnit === 'lbs' ? 'kg' : 'lbs';
    onFieldChange('weightUnit', newUnit);
  };

  const toggleHeightUnit = () => {
    const currentUnit = formData.heightUnit || 'ft';
    const newUnit = currentUnit === 'ft' ? 'cm' : 'ft';
    onFieldChange('heightUnit', newUnit);
  };

  return (
    <div className="step-container w-full h-full">
      <div className="max-w-4xl mx-auto p-6 md:p-8">
        {/* Header */}
        <div className="text-center space-y-4 mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 tracking-tight">
            Personal Information
          </h1>
          <p className="text-base md:text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Help us understand your background and create your personalized plan
          </p>
        </div>

        {/* Form Container */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm space-y-8">
          {/* Contact Information */}
          <div className="p-6 md:p-8 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <User className="h-5 w-5 text-repz-orange" />
              <h3 className="text-lg font-semibold text-gray-900">Contact Information</h3>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-3">
                <Label htmlFor="phone" className="text-sm font-medium text-gray-700">
                  Phone Number *
                </Label>
                <Input
                  id="phone"
                  type="tel"
                  placeholder="+1 (555) 123-4567"
                  value={formData.phone || ''}
                  onChange={(e) => onFieldChange('phone', e.target.value)}
                  className="h-12 px-4 border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                  required
                />
                {validation.errors.phone && (
                  <p className="text-sm text-red-500">{validation.errors.phone}</p>
                )}
              </div>

              <div className="space-y-3">
                <Label htmlFor="dateOfBirth" className="text-sm font-medium text-gray-700">
                  Date of Birth *
                </Label>
                <Input
                  id="dateOfBirth"
                  type="date"
                  value={formData.dateOfBirth || ''}
                  onChange={(e) => onFieldChange('dateOfBirth', e.target.value)}
                  className="h-12 px-4 border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                  required
                />
                {validation.errors.dateOfBirth && (
                  <p className="text-sm text-red-500">{validation.errors.dateOfBirth}</p>
                )}
                {formData.dateOfBirth && (
                  <p className="text-sm text-gray-500">
                    Age: {calculateAge(formData.dateOfBirth)} years old
                  </p>
                )}
              </div>

              <SelectField
                id="gender"
                label="Gender"
                value={formData.gender || ''}
                onChange={(value) => onFieldChange('gender', value)}
                placeholder="Select gender"
                options={[
                  { value: 'male', label: 'Male' },
                  { value: 'female', label: 'Female' },
                  { value: 'other', label: 'Other' },
                  { value: 'prefer-not-to-say', label: 'Prefer not to say' }
                ]}
              />

              <div className="space-y-3">
                <Label htmlFor="timeZone" className="text-sm font-medium text-gray-700">
                  Time Zone
                </Label>
                <Select value={formData.timeZone || 'auto'} onValueChange={(value) => onFieldChange('timeZone', value)}>
                  <SelectTrigger className="h-12 border-gray-300 focus:border-repz-orange focus:ring-repz-orange">
                    <SelectValue placeholder="Select time zone" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto-detect (PST)</SelectItem>
                    <SelectItem value="EST">Eastern Time (EST)</SelectItem>
                    <SelectItem value="CST">Central Time (CST)</SelectItem>
                    <SelectItem value="MST">Mountain Time (MST)</SelectItem>
                    <SelectItem value="PST">Pacific Time (PST)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          {/* Location */}
          <div className="p-6 md:p-8 border-t border-gray-100 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <MapPin className="h-5 w-5 text-repz-orange" />
              <h3 className="text-lg font-semibold text-gray-900">Location</h3>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-3">
                <Label htmlFor="location" className="text-sm font-medium text-gray-700">
                  Location *
                </Label>
                <Input
                  id="location"
                  type="text"
                  placeholder="City, State, Country"
                  value={formData.location || ''}
                  onChange={(e) => onFieldChange('location', e.target.value)}
                  className="h-12 px-4 border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                  required
                />
              </div>

              <div className="space-y-3">
                <Label htmlFor="preferredStartDate" className="text-sm font-medium text-gray-700">
                  Preferred Start Date
                </Label>
                <Input
                  id="preferredStartDate"
                  type="date"
                  min={new Date().toISOString().split('T')[0]}
                  value={formData.preferredStartDate || ''}
                  onChange={(e) => onFieldChange('preferredStartDate', e.target.value)}
                  className="h-12 px-4 border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                />
              </div>
            </div>
          </div>

          {/* Physical Statistics */}
          <div className="p-6 md:p-8 border-t border-gray-100 space-y-6">
            <div className="flex items-center gap-2 mb-4">
              <Calendar className="h-5 w-5 text-repz-orange" />
              <h3 className="text-lg font-semibold text-gray-900">Physical Statistics</h3>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="space-y-3">
                <Label htmlFor="currentWeight" className="text-sm font-medium text-gray-700">
                  Current Weight *
                </Label>
                <div className="flex">
                  <Input
                    id="currentWeight"
                    type="number"
                    placeholder="180"
                    value={formData.currentWeight || ''}
                    onChange={(e) => onFieldChange('currentWeight', e.target.value)}
                    className="h-12 px-4 rounded-r-none border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                    required
                  />
                  <div className="flex border border-l-0 rounded-r-md border-gray-300">
                    <button
                      type="button"
                      onClick={toggleWeightUnit}
                      className={`px-3 py-2 text-sm font-medium border-r transition-colors ${
                        (formData.weightUnit || 'lbs') === 'lbs' 
                          ? 'bg-repz-orange/10 text-repz-orange'
                          : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      lbs
                    </button>
                    <button
                      type="button"
                      onClick={toggleWeightUnit}
                      className={`px-3 py-2 text-sm font-medium transition-colors ${
                        (formData.weightUnit || 'lbs') === 'kg' 
                          ? 'bg-repz-orange/10 text-repz-orange'
                          : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      kg
                    </button>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <Label htmlFor="height" className="text-sm font-medium text-gray-700">
                  Height *
                </Label>
                <div className="flex">
                  <Input
                    id="height"
                    type="text"
                    placeholder={formData.heightUnit === 'cm' ? "175" : "5'10\""}
                    value={formData.height || ''}
                    onChange={(e) => onFieldChange('height', e.target.value)}
                    className="h-12 px-4 rounded-r-none border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                    required
                  />
                  <div className="flex border border-l-0 rounded-r-md border-gray-300">
                    <button
                      type="button"
                      onClick={toggleHeightUnit}
                      className={`px-3 py-2 text-sm font-medium border-r transition-colors ${
                        (formData.heightUnit || 'ft') === 'ft' 
                          ? 'bg-repz-orange/10 text-repz-orange'
                          : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      ft/in
                    </button>
                    <button
                      type="button"
                      onClick={toggleHeightUnit}
                      className={`px-3 py-2 text-sm font-medium transition-colors ${
                        (formData.heightUnit || 'ft') === 'cm' 
                          ? 'bg-repz-orange/10 text-repz-orange'
                          : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      cm
                    </button>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <Label htmlFor="targetWeight" className="text-sm font-medium text-gray-700">
                  Target Weight
                </Label>
                <Input
                  id="targetWeight"
                  type="number"
                  placeholder="170"
                  value={formData.targetWeight || ''}
                  onChange={(e) => onFieldChange('targetWeight', e.target.value)}
                  className="h-12 px-4 border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                />
                <p className="text-sm text-gray-500">Optional</p>
              </div>
            </div>

            <div className="mt-6">
              <div className="space-y-3">
                <Label htmlFor="bodyFatPercentage" className="text-sm font-medium text-gray-700">
                  Body Fat Percentage (if known)
                </Label>
                <Input
                  id="bodyFatPercentage"
                  type="number"
                  placeholder="15"
                  min="1"
                  max="50"
                  value={formData.bodyFatPercentage || ''}
                  onChange={(e) => onFieldChange('bodyFatPercentage', e.target.value)}
                  className="h-12 px-4 max-w-xs border-gray-300 focus:border-repz-orange focus:ring-repz-orange"
                />
                <p className="text-sm text-gray-500">
                  Optional - we can help determine this if you don't know
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};