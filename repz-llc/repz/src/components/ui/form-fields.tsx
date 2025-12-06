import React from 'react';
import { Label } from "@/ui/atoms/Label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/ui/molecules/Select";

interface FormFieldProps {
  id: string;
  label: string;
  required?: boolean;
  description?: string;
  className?: string;
}

interface TextAreaFieldProps extends FormFieldProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  rows?: number;
}

interface SelectFieldProps extends FormFieldProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  options: Array<{ value: string; label: string }>;
}

export const TextAreaField: React.FC<TextAreaFieldProps> = ({
  id,
  label,
  value,
  onChange,
  placeholder,
  required = false,
  description,
  rows = 4,
  className = ''
}) => {
  return (
    <div className="space-y-3">
      <Label htmlFor={id} className="text-sm font-medium text-gray-700">
        {label} {required && '*'}
      </Label>
      <Textarea
        id={id}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`min-h-[${rows * 30}px] border-gray-300 focus:border-repz-orange focus:ring-repz-orange ${className}`}
        required={required}
      />
      {description && (
        <p className="text-sm text-gray-500">{description}</p>
      )}
    </div>
  );
};

export const SelectField: React.FC<SelectFieldProps> = ({
  id,
  label,
  value,
  onChange,
  placeholder,
  options,
  required = false,
  description,
  className = ''
}) => {
  return (
    <div className="space-y-3">
      <Label htmlFor={id} className="text-sm font-medium text-gray-700">
        {label} {required && '*'}
      </Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className={`h-12 border-gray-300 focus:border-repz-orange focus:ring-repz-orange ${className}`}>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {description && (
        <p className="text-sm text-gray-500">{description}</p>
      )}
    </div>
  );
};