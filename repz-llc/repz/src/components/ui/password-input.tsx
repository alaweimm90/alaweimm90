import React from 'react';
import { Eye, EyeOff } from 'lucide-react';
import { Input } from "@/ui/atoms/Input";
import { Label } from "@/ui/atoms/Label";

interface PasswordInputProps {
  id?: string;
  label?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  required?: boolean;
  showPassword: boolean;
  onTogglePassword: () => void;
  className?: string;
  showStrengthIndicator?: boolean;
  passwordStrength?: string;
}

export const PasswordInput: React.FC<PasswordInputProps> = ({
  id = 'password',
  label = 'Password',
  value,
  onChange,
  placeholder = 'Enter password',
  required = false,
  showPassword,
  onTogglePassword,
  className = '',
  showStrengthIndicator = false,
  passwordStrength = ''
}) => {
  const getPasswordStrengthColor = (strength: string) => {
    switch (strength) {
      case 'weak': return 'bg-red-400';
      case 'medium': return 'bg-amber-400';
      case 'strong': return 'bg-emerald-400';
      default: return 'bg-gray-200';
    }
  };

  return (
    <div className="space-y-3">
      {label && (
        <Label htmlFor={id} className="text-sm font-medium text-gray-700">
          {label} {required && '*'}
        </Label>
      )}
      <div className="relative">
        <Input
          id={id}
          type={showPassword ? "text" : "password"}
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={`h-12 px-4 pr-12 text-base border-gray-300 focus:border-repz-orange focus:ring-repz-orange transition-colors duration-200 ${className}`}
          required={required}
        />
        <button
          type="button"
          onClick={onTogglePassword}
          className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors duration-200"
        >
          {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
        </button>
      </div>
      
      {showStrengthIndicator && value && (
        <div className="space-y-2">
          <div className="flex gap-2">
            <div className={`h-2 flex-1 rounded-full transition-all duration-300 ${getPasswordStrengthColor(passwordStrength)}`}></div>
            <div className={`h-2 flex-1 rounded-full transition-all duration-300 ${passwordStrength !== 'weak' ? getPasswordStrengthColor(passwordStrength) : 'bg-gray-200'}`}></div>
            <div className={`h-2 flex-1 rounded-full transition-all duration-300 ${passwordStrength === 'strong' ? getPasswordStrengthColor(passwordStrength) : 'bg-gray-200'}`}></div>
          </div>
          <p className="text-sm text-gray-600">
            Password strength: <span className="capitalize font-medium">{passwordStrength}</span>
          </p>
        </div>
      )}
    </div>
  );
};