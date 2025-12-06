// src/components/intake/steps/Step1Account.tsx
import React from 'react';
import { Input } from "@/ui/atoms/Input";
import { Label } from "@/ui/atoms/Label";
import { Button } from "@/ui/atoms/Button";
import { Checkbox } from "@/ui/atoms/Checkbox";
import { Shield, Eye, EyeOff } from 'lucide-react';
import { getTierConfig } from '@/constants/tiers';
import { UnifiedTierCard } from '@/components/ui/unified-tier-card';
import { PasswordInput } from '@/components/ui/password-input';
import { useFormValidation, stepValidationRules } from '@/hooks/useFormValidation';

interface Step1FormData {
  fullName?: string;
  email?: string;
  password?: string;
  confirmPassword?: string;
  termsAccepted?: boolean;
  [key: string]: string | boolean | undefined;
}

interface Step1AccountProps {
  formData: Step1FormData;
  onFieldChange: (field: string, value: string | boolean) => void;
  selectedTier: string;
  tierPrice: string;
  onChangeTier: () => void;
  showPassword: boolean;
  onTogglePassword: () => void;
  passwordStrength: string;
  tierColors?: {
    primary: string;
    background: string;
    text: string;
    accent: string;
    border: string;
    badge: string;
  };
}

export const Step1Account: React.FC<Step1AccountProps> = ({
  formData,
  onFieldChange,
  selectedTier,
  tierPrice,
  onChangeTier,
  showPassword,
  onTogglePassword,
  passwordStrength,
  tierColors
}) => {
  const { validateFormData } = useFormValidation();
  const validation = validateFormData(formData, stepValidationRules.step1);

  // Enhanced Concierge tier styling with luxury aesthetics
  const isConcierge = selectedTier.includes('Longevity Concierge') || selectedTier.includes('⚫') || selectedTier.toLowerCase().includes('concierge');
  
  const containerBg = isConcierge 
    ? 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%)' 
    : '#ffffff';
    
  const textColor = isConcierge ? '#FFD700' : '#1f2937';
  const descriptionColor = isConcierge ? '#E6C200' : '#6b7280';
  const formBg = isConcierge 
    ? 'linear-gradient(145deg, rgba(0,0,0,0.8) 0%, rgba(26,26,26,0.9) 100%)' 
    : '#ffffff';
  const labelColor = isConcierge ? '#FFD700' : '#374151';
  const inputBg = isConcierge ? 'rgba(255, 215, 0, 0.08)' : '#ffffff';
  const inputText = isConcierge ? '#F5F5F5' : '#111827';
  const inputBorder = isConcierge ? 'rgba(255, 215, 0, 0.4)' : '#d1d5db';
  const inputFocusBorder = isConcierge ? '#FFD700' : '#3b82f6';
  const buttonShadow = isConcierge ? '0 0 20px rgba(255, 215, 0, 0.3)' : '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
  const formShadow = isConcierge 
    ? '0 25px 50px -12px rgba(255, 215, 0, 0.25), inset 0 1px 0 rgba(255, 215, 0, 0.1)' 
    : '0 4px 6px -1px rgba(0, 0, 0, 0.1)';

  return (
    <div 
      className="step-container w-full h-full"
      style={{ 
        background: containerBg,
        color: textColor 
      }}
    >
      <div className="max-w-4xl mx-auto p-6 md:p-8">
        {/* Header */}
        <div className="text-center space-y-4 mb-8">
          <h1 
            className="text-3xl md:text-4xl font-bold tracking-tight"
            style={{ color: tierColors?.primary || textColor }}
          >
            Create Your Coaching Account
          </h1>
          <p 
            className="text-base md:text-lg max-w-2xl mx-auto leading-relaxed"
            style={{ color: descriptionColor }}
          >
            Set up your secure account to get started with{' '}
            <span style={{ color: tierColors?.primary || 'hsl(var(--repz-orange))' }} className="font-semibold">
              {selectedTier.replace(/^[🧑‍💻🟠⚫👑]\s*/u, '')}
            </span>{' '}
            coaching
          </p>
        </div>

        {/* Selected Tier Display */}
        <div className="mb-8">
          <UnifiedTierCard
            tier={selectedTier}
            variant="comparison"
            showFeatures={true}
            className="border-l-4 border-l-orange-500"
          />
        </div>

        {/* Form Container */}
        <div 
          className="rounded-xl shadow-sm p-6 md:p-8 space-y-8 border-2"
          style={{
            background: formBg,
            borderColor: isConcierge ? 'rgba(255, 215, 0, 0.4)' : (tierColors?.border || '#e5e7eb'),
            boxShadow: formShadow
          }}
        >
          {/* Personal Information */}
          <div className="space-y-6">
            <h3 
              className="text-lg font-semibold border-b pb-3"
              style={{ 
                color: labelColor,
                borderBottomColor: isConcierge ? '#4b5563' : '#f3f4f6'
              }}
            >
              Personal Information
            </h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Full Name */}
              <div className="space-y-3">
                <Label 
                  htmlFor="fullName" 
                  className="text-sm font-medium"
                  style={{ color: labelColor }}
                >
                  Full Name *
                </Label>
                <Input
                  id="fullName"
                  type="text"
                  placeholder="Enter your full name"
                  value={formData.fullName || ''}
                  onChange={(e) => onFieldChange('fullName', e.target.value)}
                  className="h-12 px-4 text-base transition-colors duration-200"
                  style={{
                    backgroundColor: inputBg,
                    color: inputText,
                    borderColor: inputBorder,
                    '--focus-border': tierColors?.primary || 'hsl(var(--repz-orange))',
                    '--focus-ring': tierColors?.primary || 'hsl(var(--repz-orange))'
                  } as React.CSSProperties & { [key: string]: string }}
                  onFocus={(e) => {
                    e.target.style.borderColor = tierColors?.primary || 'hsl(var(--repz-orange))';
                    e.target.style.boxShadow = tierColors?.primary ? `0 0 0 3px ${tierColors.primary}20` : '0 0 0 3px hsl(var(--repz-orange) / 0.2)';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = inputBorder;
                    e.target.style.boxShadow = 'none';
                  }}
                  required
                />
                {validation.errors.fullName && (
                  <p className="text-sm text-red-500 mt-1">{validation.errors.fullName}</p>
                )}
              </div>

              {/* Email */}
              <div className="space-y-3">
                <Label 
                  htmlFor="email" 
                  className="text-sm font-medium"
                  style={{ color: labelColor }}
                >
                  Email Address *
                </Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="your@email.com"
                  value={formData.email || ''}
                  onChange={(e) => onFieldChange('email', e.target.value)}
                  className="h-12 px-4 text-base transition-colors duration-200"
                  style={{
                    backgroundColor: inputBg,
                    color: inputText,
                    borderColor: inputBorder
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = tierColors?.primary || 'hsl(var(--repz-orange))';
                    e.target.style.boxShadow = tierColors?.primary ? `0 0 0 3px ${tierColors.primary}20` : '0 0 0 3px hsl(var(--repz-orange) / 0.2)';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = inputBorder;
                    e.target.style.boxShadow = 'none';
                  }}
                  required
                />
                {validation.errors.email && (
                  <p className="text-sm text-red-500 mt-1">{validation.errors.email}</p>
                )}
                <p 
                  className="text-sm"
                  style={{ color: descriptionColor }}
                >
                  This will be your login email address
                </p>
              </div>
            </div>
          </div>

          {/* Account Security */}
          <div className="space-y-6">
            <h3 
              className="text-lg font-semibold border-b pb-3"
              style={{ 
                color: labelColor,
                borderBottomColor: isConcierge ? '#4b5563' : '#f3f4f6'
              }}
            >
              Account Security
            </h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Password */}
              <div 
                className="space-y-3 tier-password-wrapper"
                style={{
                  '--input-bg': inputBg,
                  '--input-text': inputText,
                  '--input-border': inputBorder,
                  '--tier-primary': tierColors?.primary || 'hsl(var(--repz-orange))'
                } as React.CSSProperties & { [key: string]: string }}
              >
                <Label 
                  htmlFor="password" 
                  className="text-sm font-medium"
                  style={{ color: labelColor }}
                >
                  Create Password *
                </Label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Create a secure password"
                    value={formData.password || ''}
                    onChange={(e) => onFieldChange('password', e.target.value)}
                    className="h-12 px-4 pr-12 text-base transition-colors duration-200"
                    style={{
                      backgroundColor: inputBg,
                      color: inputText,
                      borderColor: inputBorder
                    }}
                    onFocus={(e) => {
                      e.target.style.borderColor = tierColors?.primary || 'hsl(var(--repz-orange))';
                      e.target.style.boxShadow = `0 0 0 3px ${tierColors?.primary}20` || '0 0 0 3px hsl(var(--repz-orange) / 0.2)';
                    }}
                    onBlur={(e) => {
                      e.target.style.borderColor = inputBorder;
                      e.target.style.boxShadow = 'none';
                    }}
                    required
                  />
                  <button
                    type="button"
                    onClick={onTogglePassword}
                    className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors duration-200"
                  >
                    {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                  </button>
                </div>
                {passwordStrength && formData.password && (
                  <div className="space-y-2">
                    <div className="flex gap-2">
                      <div className={`h-2 flex-1 rounded-full transition-all duration-300 ${
                        passwordStrength === 'weak' ? 'bg-red-400' :
                        passwordStrength === 'medium' ? 'bg-amber-400' :
                        passwordStrength === 'strong' ? 'bg-emerald-400' : 'bg-gray-200'
                      }`}></div>
                      <div className={`h-2 flex-1 rounded-full transition-all duration-300 ${
                        passwordStrength === 'medium' || passwordStrength === 'strong' ? 
                          (passwordStrength === 'medium' ? 'bg-amber-400' : 'bg-emerald-400') : 'bg-gray-200'
                      }`}></div>
                      <div className={`h-2 flex-1 rounded-full transition-all duration-300 ${
                        passwordStrength === 'strong' ? 'bg-emerald-400' : 'bg-gray-200'
                      }`}></div>
                    </div>
                    <p 
                      className="text-sm"
                      style={{ color: labelColor }}
                    >
                      Password strength: <span className="capitalize font-medium">{passwordStrength}</span>
                    </p>
                  </div>
                )}
                {validation.errors.password && (
                  <p className="text-sm text-red-500 mt-1">{validation.errors.password}</p>
                )}
              </div>

              {/* Confirm Password */}
              <div className="space-y-3">
                <Label 
                  htmlFor="confirmPassword" 
                  className="text-sm font-medium"
                  style={{ color: labelColor }}
                >
                  Confirm Password *
                </Label>
                <div className="relative">
                  <Input
                    id="confirmPassword"
                    type="password"
                    placeholder="Confirm your password"
                    value={formData.confirmPassword || ''}
                    onChange={(e) => onFieldChange('confirmPassword', e.target.value)}
                    className="h-12 px-4 text-base transition-colors duration-200"
                    style={{
                      backgroundColor: inputBg,
                      color: inputText,
                      borderColor: inputBorder
                    }}
                    onFocus={(e) => {
                      e.target.style.borderColor = tierColors?.primary || 'hsl(var(--repz-orange))';
                      e.target.style.boxShadow = `0 0 0 3px ${tierColors?.primary}20` || '0 0 0 3px hsl(var(--repz-orange) / 0.2)';
                    }}
                    onBlur={(e) => {
                      e.target.style.borderColor = inputBorder;
                      e.target.style.boxShadow = 'none';
                    }}
                    required
                  />
                </div>
                {validation.errors.confirmPassword && (
                  <p className="text-sm text-red-500 mt-1">{validation.errors.confirmPassword}</p>
                )}
              </div>
            </div>
          </div>

          {/* Terms and Security */}
          <div className="space-y-6 pt-4">
            <div className="space-y-2">
              <div className="flex items-start space-x-4">
                <div 
                  className="tier-checkbox-wrapper"
                  style={{
                    '--tier-primary': tierColors?.primary || 'hsl(var(--repz-orange))'
                  } as React.CSSProperties & { [key: string]: string }}
                >
                  <Checkbox
                    id="termsAccepted"
                    checked={formData.termsAccepted || false}
                    onCheckedChange={(checked) => onFieldChange('termsAccepted', checked)}
                    className="mt-1 h-5 w-5 tier-checkbox data-[state=checked]:text-white"
                  />
                </div>
                <Label 
                  htmlFor="termsAccepted" 
                  className="text-sm leading-relaxed"
                  style={{ color: labelColor }}
                >
                  I agree to the{' '}
                  <a 
                    href="/terms" 
                    target="_blank" 
                    className="underline font-medium transition-colors duration-200"
                    style={{ color: tierColors?.primary || 'hsl(var(--repz-orange))' }}
                  >
                    Terms of Service
                  </a>{' '}
                  and{' '}
                  <a 
                    href="/privacy" 
                    target="_blank" 
                    className="underline font-medium transition-colors duration-200"
                    style={{ color: tierColors?.primary || 'hsl(var(--repz-orange))' }}
                  >
                    Privacy Policy
                  </a>
                </Label>
              </div>
              {validation.errors.termsAccepted && (
                <p className="text-sm text-red-500 mt-1">{validation.errors.termsAccepted}</p>
              )}
            </div>

            {/* Security Notice */}
            <div 
              className="border rounded-lg p-6"
              style={{
                backgroundColor: isConcierge ? '#374151' : '#f8fafc',
                borderColor: isConcierge ? '#6b7280' : '#e2e8f0'
              }}
            >
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0">
                  <Shield 
                    className="h-6 w-6" 
                    style={{ color: tierColors?.primary || '#3b82f6' }}
                  />
                </div>
                <div className="space-y-2">
                  <h4 
                    className="font-semibold text-base"
                    style={{ color: tierColors?.primary || '#1e40af' }}
                  >
                    Secure Account Creation
                  </h4>
                  <p 
                    className="text-sm leading-relaxed"
                    style={{ color: labelColor }}
                  >
                    Your account will be encrypted and secure. You'll use this email and password to access your coaching dashboard and track your progress.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};