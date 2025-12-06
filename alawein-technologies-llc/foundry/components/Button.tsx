/**
 * Button Component
 * A flexible, accessible button component with multiple variants and sizes
 */

import React, { forwardRef, ButtonHTMLAttributes } from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { Loader2 } from 'lucide-react';
import { cn } from '../utils/cn';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, HTMLMotionProps<'button'> {
  /**
   * Button variant style
   */
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'ghost' | 'outline' | 'link';

  /**
   * Button size
   */
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';

  /**
   * Full width button
   */
  fullWidth?: boolean;

  /**
   * Loading state
   */
  loading?: boolean;

  /**
   * Loading text to display
   */
  loadingText?: string;

  /**
   * Icon to display before text
   */
  leftIcon?: React.ReactNode;

  /**
   * Icon to display after text
   */
  rightIcon?: React.ReactNode;

  /**
   * Rounded style
   */
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'xl' | 'full';

  /**
   * Additional class names
   */
  className?: string;

  /**
   * Children elements
   */
  children?: React.ReactNode;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      fullWidth = false,
      loading = false,
      loadingText = 'Loading...',
      leftIcon,
      rightIcon,
      rounded = 'md',
      className,
      children,
      disabled,
      type = 'button',
      ...props
    },
    ref
  ) => {
    // Variant styles
    const variants = {
      primary: `
        bg-primary-600 text-white
        hover:bg-primary-700
        focus:ring-primary-500
        disabled:bg-primary-300
        dark:bg-primary-500 dark:hover:bg-primary-600 dark:disabled:bg-primary-800
      `,
      secondary: `
        bg-secondary-600 text-white
        hover:bg-secondary-700
        focus:ring-secondary-500
        disabled:bg-secondary-300
        dark:bg-secondary-500 dark:hover:bg-secondary-600 dark:disabled:bg-secondary-800
      `,
      success: `
        bg-success-600 text-white
        hover:bg-success-700
        focus:ring-success-500
        disabled:bg-success-300
        dark:bg-success-500 dark:hover:bg-success-600 dark:disabled:bg-success-800
      `,
      warning: `
        bg-warning-500 text-white
        hover:bg-warning-600
        focus:ring-warning-400
        disabled:bg-warning-300
        dark:bg-warning-400 dark:hover:bg-warning-500 dark:disabled:bg-warning-700
      `,
      danger: `
        bg-danger-600 text-white
        hover:bg-danger-700
        focus:ring-danger-500
        disabled:bg-danger-300
        dark:bg-danger-500 dark:hover:bg-danger-600 dark:disabled:bg-danger-800
      `,
      ghost: `
        bg-transparent text-gray-700
        hover:bg-gray-100
        focus:ring-gray-400
        disabled:text-gray-300
        dark:text-gray-200 dark:hover:bg-gray-800 dark:disabled:text-gray-600
      `,
      outline: `
        bg-transparent text-gray-700
        border border-gray-300
        hover:bg-gray-50
        focus:ring-gray-400
        disabled:text-gray-300 disabled:border-gray-200
        dark:text-gray-200 dark:border-gray-600 dark:hover:bg-gray-800 dark:disabled:text-gray-600 dark:disabled:border-gray-700
      `,
      link: `
        bg-transparent text-primary-600
        hover:text-primary-700 hover:underline
        focus:ring-primary-500
        disabled:text-primary-300
        dark:text-primary-400 dark:hover:text-primary-300 dark:disabled:text-primary-700
      `,
    };

    // Size styles
    const sizes = {
      xs: 'px-2.5 py-1.5 text-xs',
      sm: 'px-3 py-2 text-sm',
      md: 'px-4 py-2.5 text-base',
      lg: 'px-5 py-3 text-lg',
      xl: 'px-6 py-3.5 text-xl',
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

    // Base classes
    const baseClasses = `
      inline-flex items-center justify-center
      font-medium transition-all duration-200
      focus:outline-none focus:ring-2 focus:ring-offset-2
      disabled:cursor-not-allowed disabled:opacity-50
      relative overflow-hidden
    `;

    // Combine all classes
    const buttonClasses = cn(
      baseClasses,
      variants[variant],
      sizes[size],
      roundedStyles[rounded],
      fullWidth && 'w-full',
      className
    );

    return (
      <motion.button
        ref={ref}
        type={type}
        disabled={disabled || loading}
        className={buttonClasses}
        whileHover={{ scale: disabled || loading ? 1 : 1.02 }}
        whileTap={{ scale: disabled || loading ? 1 : 0.98 }}
        transition={{ duration: 0.1 }}
        {...props}
      >
        {/* Loading spinner */}
        {loading && (
          <Loader2
            className={cn(
              'animate-spin',
              children || loadingText ? 'mr-2' : '',
              size === 'xs' && 'h-3 w-3',
              size === 'sm' && 'h-4 w-4',
              size === 'md' && 'h-5 w-5',
              size === 'lg' && 'h-6 w-6',
              size === 'xl' && 'h-7 w-7'
            )}
          />
        )}

        {/* Left icon */}
        {!loading && leftIcon && (
          <span className={cn(children && 'mr-2', 'inline-flex items-center')}>
            {leftIcon}
          </span>
        )}

        {/* Button text */}
        {loading ? loadingText || children : children}

        {/* Right icon */}
        {!loading && rightIcon && (
          <span className={cn(children && 'ml-2', 'inline-flex items-center')}>
            {rightIcon}
          </span>
        )}

        {/* Ripple effect on click */}
        <motion.span
          className="absolute inset-0 bg-white/20"
          initial={{ scale: 0, opacity: 0.5 }}
          animate={{ scale: 0, opacity: 0.5 }}
          whileTap={{ scale: 2, opacity: 0 }}
          transition={{ duration: 0.6 }}
          style={{ borderRadius: '50%' }}
        />
      </motion.button>
    );
  }
);

Button.displayName = 'Button';

/**
 * Button Group Component
 * Groups multiple buttons together
 */
export interface ButtonGroupProps {
  /**
   * Direction of button group
   */
  direction?: 'horizontal' | 'vertical';

  /**
   * Size of all buttons in group
   */
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';

  /**
   * Variant for all buttons in group
   */
  variant?: ButtonProps['variant'];

  /**
   * Additional class names
   */
  className?: string;

  /**
   * Children buttons
   */
  children: React.ReactNode;
}

export const ButtonGroup: React.FC<ButtonGroupProps> = ({
  direction = 'horizontal',
  size,
  variant,
  className,
  children,
}) => {
  const baseClasses = 'inline-flex';
  const directionClasses = {
    horizontal: 'flex-row',
    vertical: 'flex-col',
  };

  const groupClasses = cn(
    baseClasses,
    directionClasses[direction],
    className
  );

  // Clone children and pass size/variant props
  const enhancedChildren = React.Children.map(children, (child, index) => {
    if (React.isValidElement(child)) {
      const isFirst = index === 0;
      const isLast = index === React.Children.count(children) - 1;

      let roundedClass = '';
      if (direction === 'horizontal') {
        if (isFirst && !isLast) roundedClass = 'rounded-r-none';
        else if (isLast && !isFirst) roundedClass = 'rounded-l-none';
        else if (!isFirst && !isLast) roundedClass = 'rounded-none';
      } else {
        if (isFirst && !isLast) roundedClass = 'rounded-b-none';
        else if (isLast && !isFirst) roundedClass = 'rounded-t-none';
        else if (!isFirst && !isLast) roundedClass = 'rounded-none';
      }

      return React.cloneElement(child as React.ReactElement<ButtonProps>, {
        size: size || (child.props as ButtonProps).size,
        variant: variant || (child.props as ButtonProps).variant,
        className: cn((child.props as ButtonProps).className, roundedClass),
      });
    }
    return child;
  });

  return <div className={groupClasses}>{enhancedChildren}</div>;
};

ButtonGroup.displayName = 'ButtonGroup';

export default Button;