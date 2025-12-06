import React from 'react';
import { FieldErrors, UseFormRegister, RegisterOptions } from 'react-hook-form';
import { AlertCircle, CheckCircle2, Eye, EyeOff } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface ValidationInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  name: string;
  label?: string;
  register: UseFormRegister<any>;
  errors?: FieldErrors;
  validation?: RegisterOptions;
  showPasswordToggle?: boolean;
  showValidationIcon?: boolean;
  helperText?: string;
}

export const ValidationInput: React.FC<ValidationInputProps> = ({
  name,
  label,
  register,
  errors,
  validation,
  showPasswordToggle = false,
  showValidationIcon = true,
  helperText,
  className,
  type = 'text',
  ...props
}) => {
  const [showPassword, setShowPassword] = React.useState(false);
  const error = errors?.[name as keyof typeof errors];
  const hasError = !!error;
  const isValid = !hasError && props.value && String(props.value).length > 0;

  const inputType = showPasswordToggle && showPassword ? 'text' : type;

  return (
    <div className="space-y-2">
      {label && (
        <label htmlFor={name} className="block text-sm font-medium text-gray-700">
          {label}
        </label>
      )}
      <div className="relative">
        <input
          id={name}
          type={inputType}
          className={cn(
            'block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm',
            hasError && 'border-red-300 focus:border-red-500 focus:ring-red-500',
            isValid && 'border-green-300 focus:border-green-500 focus:ring-green-500',
            className
          )}
          {...register(name, validation)}
          {...props}
        />
        
        {showValidationIcon && (
          <div className="absolute inset-y-0 right-0 flex items-center pr-3">
            {hasError && <AlertCircle className="h-4 w-4 text-red-500" />}
            {isValid && <CheckCircle2 className="h-4 w-4 text-green-500" />}
          </div>
        )}

        {showPasswordToggle && type === 'password' && (
          <button
            type="button"
            className="absolute inset-y-0 right-0 flex items-center pr-3"
            onClick={() => setShowPassword(!showPassword)}
          >
            {showPassword ? (
              <EyeOff className="h-4 w-4 text-gray-400" />
            ) : (
              <Eye className="h-4 w-4 text-gray-400" />
            )}
          </button>
        )}
      </div>

      {hasError && (
        <p className="text-sm text-red-600">
          {error.message as string}
        </p>
      )}

      {helperText && !hasError && (
        <p className="text-sm text-gray-500">
          {helperText}
        </p>
      )}
    </div>
  );
};

export interface ValidationSelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  name: string;
  label?: string;
  register: UseFormRegister<any>;
  errors?: FieldErrors;
  validation?: RegisterOptions;
  options: Array<{ value: string; label: string }>;
  placeholder?: string;
  helperText?: string;
}

export const ValidationSelect: React.FC<ValidationSelectProps> = ({
  name,
  label,
  register,
  errors,
  validation,
  options,
  placeholder,
  helperText,
  className,
  ...props
}) => {
  const error = errors?.[name as keyof typeof errors];
  const hasError = !!error;

  return (
    <div className="space-y-2">
      {label && (
        <label htmlFor={name} className="block text-sm font-medium text-gray-700">
          {label}
        </label>
      )}
      <select
        id={name}
        className={cn(
          'block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm',
          hasError && 'border-red-300 focus:border-red-500 focus:ring-red-500',
          className
        )}
        {...register(name, validation)}
        {...props}
      >
        {placeholder && (
          <option value="">{placeholder}</option>
        )}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>

      {hasError && (
        <p className="text-sm text-red-600">
          {error.message as string}
        </p>
      )}

      {helperText && !hasError && (
        <p className="text-sm text-gray-500">
          {helperText}
        </p>
      )}
    </div>
  );
};

export interface ValidationTextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  name: string;
  label?: string;
  register: UseFormRegister<any>;
  errors?: FieldErrors;
  validation?: RegisterOptions;
  helperText?: string;
  rows?: number;
}

export const ValidationTextarea: React.FC<ValidationTextareaProps> = ({
  name,
  label,
  register,
  errors,
  validation,
  helperText,
  rows = 4,
  className,
  ...props
}) => {
  const error = errors?.[name as keyof typeof errors];
  const hasError = !!error;

  return (
    <div className="space-y-2">
      {label && (
        <label htmlFor={name} className="block text-sm font-medium text-gray-700">
          {label}
        </label>
      )}
      <textarea
        id={name}
        rows={rows}
        className={cn(
          'block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm',
          hasError && 'border-red-300 focus:border-red-500 focus:ring-red-500',
          className
        )}
        {...register(name, validation)}
        {...props}
      />

      {hasError && (
        <p className="text-sm text-red-600">
          {error.message as string}
        </p>
      )}

      {helperText && !hasError && (
        <p className="text-sm text-gray-500">
          {helperText}
        </p>
      )}
    </div>
  );
};

export interface ValidationCheckboxProps extends React.InputHTMLAttributes<HTMLInputElement> {
  name: string;
  label?: string;
  register: UseFormRegister<any>;
  errors?: FieldErrors;
  validation?: RegisterOptions;
  helperText?: string;
}

export const ValidationCheckbox: React.FC<ValidationCheckboxProps> = ({
  name,
  label,
  register,
  errors,
  validation,
  helperText,
  className,
  ...props
}) => {
  const error = errors?.[name as keyof typeof errors];
  const hasError = !!error;

  return (
    <div className="space-y-2">
      <div className="flex items-center">
        <input
          type="checkbox"
          id={name}
          className={cn(
            'h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500',
            hasError && 'border-red-300 text-red-600 focus:ring-red-500',
            className
          )}
          {...register(name, validation)}
          {...props}
        />
        {label && (
          <label htmlFor={name} className="ml-2 block text-sm text-gray-700">
            {label}
          </label>
        )}
      </div>

      {hasError && (
        <p className="text-sm text-red-600">
          {error.message as string}
        </p>
      )}

      {helperText && !hasError && (
        <p className="text-sm text-gray-500">
          {helperText}
        </p>
      )}
    </div>
  );
};

export interface PasswordStrengthIndicatorProps {
  strength: {
    score: number;
    level: 'weak' | 'medium' | 'strong';
    feedback: string[];
  };
  showFeedback?: boolean;
}

export const PasswordStrengthIndicator: React.FC<PasswordStrengthIndicatorProps> = ({
  strength,
  showFeedback = true
}) => {
  const getStrengthColor = () => {
    switch (strength.level) {
      case 'weak':
        return 'bg-red-500';
      case 'medium':
        return 'bg-yellow-500';
      case 'strong':
        return 'bg-green-500';
      default:
        return 'bg-gray-300';
    }
  };

  const getStrengthText = () => {
    switch (strength.level) {
      case 'weak':
        return 'Weak';
      case 'medium':
        return 'Medium';
      case 'strong':
        return 'Strong';
      default:
        return '';
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center space-x-2">
        <div className="flex-1 bg-gray-200 rounded-full h-2">
          <div
            className={cn(getStrengthColor(), 'h-2 rounded-full transition-all duration-300')}
            style={{ width: `${(strength.score / 6) * 100}%` }}
          />
        </div>
        <span className="text-sm font-medium text-gray-600">
          {getStrengthText()}
        </span>
      </div>

      {showFeedback && strength.feedback.length > 0 && (
        <ul className="text-xs text-gray-500 space-y-1">
          {strength.feedback.map((item, index) => (
            <li key={index} className="flex items-center space-x-1">
              <span>•</span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export interface FormErrorSummaryProps {
  errors: FieldErrors;
  className?: string;
}

export const FormErrorSummary: React.FC<FormErrorSummaryProps> = ({ errors, className }) => {
  const errorMessages = Object.entries(errors).map(([field, error]) => {
    const fieldName = field
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, str => str.toUpperCase())
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .toLowerCase()
      .replace(/^./, str => str.toUpperCase());

    return {
      field: fieldName,
      message: error?.message as string
    };
  });

  if (errorMessages.length === 0) {
    return null;
  }

  return (
    <div className={cn('rounded-md bg-red-50 p-4', className)}>
      <div className="flex">
        <div className="flex-shrink-0">
          <AlertCircle className="h-5 w-5 text-red-400" />
        </div>
        <div className="ml-3">
          <h3 className="text-sm font-medium text-red-800">
            Please correct the following errors:
          </h3>
          <div className="mt-2 text-sm text-red-700">
            <ul className="list-disc space-y-1 pl-5">
              {errorMessages.map((error, index) => (
                <li key={index}>
                  <span className="font-medium">{error.field}:</span> {error.message}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export interface FormSuccessMessageProps {
  message: string;
  className?: string;
  onDismiss?: () => void;
}

export const FormSuccessMessage: React.FC<FormSuccessMessageProps> = ({ 
  message, 
  className, 
  onDismiss 
}) => {
  return (
    <div className={cn('rounded-md bg-green-50 p-4', className)}>
      <div className="flex">
        <div className="flex-shrink-0">
          <CheckCircle2 className="h-5 w-5 text-green-400" />
        </div>
        <div className="ml-3">
          <p className="text-sm font-medium text-green-800">{message}</p>
        </div>
        {onDismiss && (
          <div className="ml-auto pl-3">
            <div className="-mx-1.5 -my-1.5">
              <button
                type="button"
                className="inline-flex rounded-md bg-green-50 p-1.5 text-green-500 hover:bg-green-100 focus:outline-none focus:ring-2 focus:ring-green-600 focus:ring-offset-2 focus:ring-offset-green-50"
                onClick={onDismiss}
              >
                <span className="sr-only">Dismiss</span>
                <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path
                    fillRule="evenodd"
                    d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};