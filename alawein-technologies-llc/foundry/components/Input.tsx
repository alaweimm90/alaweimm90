/**
 * Input Component
 * A comprehensive input field component with various states and validation support
 */

import React, { forwardRef, InputHTMLAttributes, useState } from 'react';
import { motion } from 'framer-motion';
import { Eye, EyeOff, AlertCircle, CheckCircle, Info, X } from 'lucide-react';
import { cn } from '../utils/cn';

export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  /**
   * Input size
   */
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';

  /**
   * Input variant
   */
  variant?: 'outline' | 'filled' | 'flushed' | 'unstyled';

  /**
   * Full width input
   */
  fullWidth?: boolean;

  /**
   * Label for the input
   */
  label?: string;

  /**
   * Helper text below input
   */
  helperText?: string;

  /**
   * Error state and message
   */
  error?: boolean | string;

  /**
   * Success state and message
   */
  success?: boolean | string;

  /**
   * Warning state and message
   */
  warning?: boolean | string;

  /**
   * Info state and message
   */
  info?: boolean | string;

  /**
   * Icon to display before input
   */
  leftIcon?: React.ReactNode;

  /**
   * Icon to display after input
   */
  rightIcon?: React.ReactNode;

  /**
   * Show clear button when input has value
   */
  clearable?: boolean;

  /**
   * Loading state
   */
  loading?: boolean;

  /**
   * Rounded style
   */
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'xl' | 'full';

  /**
   * On clear callback
   */
  onClear?: () => void;

  /**
   * Additional wrapper class names
   */
  wrapperClassName?: string;

  /**
   * Additional class names
   */
  className?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      size = 'md',
      variant = 'outline',
      fullWidth = false,
      label,
      helperText,
      error,
      success,
      warning,
      info,
      leftIcon,
      rightIcon,
      clearable,
      loading,
      rounded = 'md',
      onClear,
      wrapperClassName,
      className,
      disabled,
      type = 'text',
      required,
      ...props
    },
    ref
  ) => {
    const [showPassword, setShowPassword] = useState(false);
    const [isFocused, setIsFocused] = useState(false);
    const isPassword = type === 'password';

    // Size styles
    const sizes = {
      xs: 'px-2.5 py-1.5 text-xs',
      sm: 'px-3 py-2 text-sm',
      md: 'px-4 py-2.5 text-base',
      lg: 'px-5 py-3 text-lg',
      xl: 'px-6 py-3.5 text-xl',
    };

    // Variant styles
    const variants = {
      outline: `
        bg-white dark:bg-gray-900
        border border-gray-300 dark:border-gray-700
        hover:border-gray-400 dark:hover:border-gray-600
        focus:border-primary-500 dark:focus:border-primary-400
      `,
      filled: `
        bg-gray-100 dark:bg-gray-800
        border border-transparent
        hover:bg-gray-200 dark:hover:bg-gray-700
        focus:bg-white dark:focus:bg-gray-900
        focus:border-primary-500 dark:focus:border-primary-400
      `,
      flushed: `
        bg-transparent
        border-0 border-b-2 border-gray-300 dark:border-gray-700
        hover:border-gray-400 dark:hover:border-gray-600
        focus:border-primary-500 dark:focus:border-primary-400
        rounded-none px-0
      `,
      unstyled: `
        bg-transparent
        border-0
        p-0
      `,
    };

    // Rounded styles
    const roundedStyles = {
      none: 'rounded-none',
      sm: 'rounded-sm',
      md: 'rounded-md',
      lg: 'rounded-lg',
      xl: 'rounded-xl',
      full: 'rounded-full',
    };

    // State styles
    const stateStyles = {
      error: 'border-danger-500 dark:border-danger-400 focus:border-danger-500 dark:focus:border-danger-400 focus:ring-danger-500',
      success: 'border-success-500 dark:border-success-400 focus:border-success-500 dark:focus:border-success-400 focus:ring-success-500',
      warning: 'border-warning-500 dark:border-warning-400 focus:border-warning-500 dark:focus:border-warning-400 focus:ring-warning-500',
      info: 'border-primary-500 dark:border-primary-400 focus:border-primary-500 dark:focus:border-primary-400 focus:ring-primary-500',
    };

    // Determine current state
    const currentState = error ? 'error' : success ? 'success' : warning ? 'warning' : info ? 'info' : null;

    // Base classes
    const baseClasses = `
      w-full
      text-gray-900 dark:text-gray-100
      placeholder-gray-400 dark:placeholder-gray-500
      transition-all duration-200
      focus:outline-none focus:ring-2 focus:ring-offset-0
      disabled:cursor-not-allowed disabled:opacity-50
    `;

    // Input classes
    const inputClasses = cn(
      baseClasses,
      variants[variant],
      sizes[size],
      variant !== 'flushed' && roundedStyles[rounded],
      currentState && stateStyles[currentState],
      leftIcon && 'pl-10',
      (rightIcon || clearable || isPassword || loading) && 'pr-10',
      className
    );

    // Wrapper classes
    const wrapperClasses = cn(
      'relative',
      fullWidth ? 'w-full' : 'inline-flex',
      wrapperClassName
    );

    // State message
    const stateMessage =
      typeof error === 'string' ? error :
      typeof success === 'string' ? success :
      typeof warning === 'string' ? warning :
      typeof info === 'string' ? info :
      helperText;

    // State icon
    const StateIcon =
      error ? AlertCircle :
      success ? CheckCircle :
      warning ? AlertCircle :
      info ? Info :
      null;

    return (
      <div className={wrapperClasses}>
        {/* Label */}
        {label && (
          <label className="block mb-1.5 text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
            {required && <span className="ml-1 text-danger-500">*</span>}
          </label>
        )}

        {/* Input container */}
        <div className="relative">
          {/* Left icon */}
          {leftIcon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none">
              <span className="text-gray-400 dark:text-gray-500">
                {leftIcon}
              </span>
            </div>
          )}

          {/* Input field */}
          <input
            ref={ref}
            type={isPassword && showPassword ? 'text' : type}
            disabled={disabled || loading}
            required={required}
            className={inputClasses}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            {...props}
          />

          {/* Right side icons container */}
          <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
            {/* Loading spinner */}
            {loading && (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                className="text-gray-400 dark:text-gray-500"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
              </motion.div>
            )}

            {/* Clear button */}
            {clearable && props.value && !loading && (
              <button
                type="button"
                onClick={onClear}
                className="text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            )}

            {/* Password toggle */}
            {isPassword && !loading && (
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 transition-colors"
              >
                {showPassword ? (
                  <EyeOff className="w-5 h-5" />
                ) : (
                  <Eye className="w-5 h-5" />
                )}
              </button>
            )}

            {/* Right icon */}
            {rightIcon && !clearable && !isPassword && !loading && (
              <span className="text-gray-400 dark:text-gray-500">
                {rightIcon}
              </span>
            )}
          </div>

          {/* Focus line for flushed variant */}
          {variant === 'flushed' && (
            <motion.div
              className="absolute bottom-0 left-0 h-0.5 bg-primary-500"
              initial={{ width: 0 }}
              animate={{ width: isFocused ? '100%' : 0 }}
              transition={{ duration: 0.2 }}
            />
          )}
        </div>

        {/* Helper/State message */}
        {stateMessage && (
          <motion.div
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn(
              'mt-1.5 text-sm flex items-center gap-1',
              error && 'text-danger-500 dark:text-danger-400',
              success && 'text-success-500 dark:text-success-400',
              warning && 'text-warning-500 dark:text-warning-400',
              info && 'text-primary-500 dark:text-primary-400',
              !currentState && 'text-gray-500 dark:text-gray-400'
            )}
          >
            {StateIcon && <StateIcon className="w-4 h-4" />}
            <span>{stateMessage}</span>
          </motion.div>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;