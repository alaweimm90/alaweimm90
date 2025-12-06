/**
 * Card Component
 * A flexible container component for grouping and displaying content
 */

import React, { forwardRef, HTMLAttributes } from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { cn } from '../utils/cn';

export interface CardProps extends HTMLAttributes<HTMLDivElement>, HTMLMotionProps<'div'> {
  /**
   * Card variant style
   */
  variant?: 'elevated' | 'outlined' | 'filled' | 'ghost';

  /**
   * Padding size
   */
  padding?: 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl';

  /**
   * Shadow size
   */
  shadow?: 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';

  /**
   * Rounded corners
   */
  rounded?: 'none' | 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full';

  /**
   * Hover effect
   */
  hoverable?: boolean;

  /**
   * Clickable card
   */
  clickable?: boolean;

  /**
   * Full width
   */
  fullWidth?: boolean;

  /**
   * Background color
   */
  bgColor?: 'white' | 'gray' | 'primary' | 'secondary' | 'transparent';

  /**
   * Border color (for outlined variant)
   */
  borderColor?: 'gray' | 'primary' | 'secondary' | 'success' | 'warning' | 'danger';

  /**
   * Additional class names
   */
  className?: string;

  /**
   * Children elements
   */
  children?: React.ReactNode;
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  (
    {
      variant = 'elevated',
      padding = 'md',
      shadow = 'md',
      rounded = 'lg',
      hoverable = false,
      clickable = false,
      fullWidth = false,
      bgColor = 'white',
      borderColor = 'gray',
      className,
      children,
      ...props
    },
    ref
  ) => {
    // Variant styles
    const variants = {
      elevated: `
        bg-white dark:bg-gray-800
        ${shadow !== 'none' ? `shadow-${shadow} dark:shadow-dark-${shadow}` : ''}
      `,
      outlined: `
        bg-white dark:bg-gray-800
        border
      `,
      filled: `
        ${bgColor === 'gray' ? 'bg-gray-100 dark:bg-gray-900' : ''}
        ${bgColor === 'primary' ? 'bg-primary-50 dark:bg-primary-950' : ''}
        ${bgColor === 'secondary' ? 'bg-secondary-50 dark:bg-secondary-950' : ''}
      `,
      ghost: `
        bg-transparent
      `,
    };

    // Border color styles
    const borderColors = {
      gray: 'border-gray-200 dark:border-gray-700',
      primary: 'border-primary-200 dark:border-primary-700',
      secondary: 'border-secondary-200 dark:border-secondary-700',
      success: 'border-success-200 dark:border-success-700',
      warning: 'border-warning-200 dark:border-warning-700',
      danger: 'border-danger-200 dark:border-danger-700',
    };

    // Background color styles
    const bgColors = {
      white: variant !== 'ghost' ? 'bg-white dark:bg-gray-800' : '',
      gray: 'bg-gray-100 dark:bg-gray-900',
      primary: 'bg-primary-50 dark:bg-primary-950',
      secondary: 'bg-secondary-50 dark:bg-secondary-950',
      transparent: 'bg-transparent',
    };

    // Padding styles
    const paddings = {
      none: '',
      xs: 'p-2',
      sm: 'p-3',
      md: 'p-4',
      lg: 'p-6',
      xl: 'p-8',
    };

    // Rounded styles
    const roundedStyles = {
      none: 'rounded-none',
      sm: 'rounded-sm',
      md: 'rounded-md',
      lg: 'rounded-lg',
      xl: 'rounded-xl',
      '2xl': 'rounded-2xl',
      full: 'rounded-full',
    };

    // Shadow styles
    const shadowStyles = {
      none: '',
      xs: 'shadow-xs dark:shadow-dark-xs',
      sm: 'shadow-sm dark:shadow-dark-sm',
      md: 'shadow-md dark:shadow-dark-md',
      lg: 'shadow-lg dark:shadow-dark-lg',
      xl: 'shadow-xl dark:shadow-dark-xl',
      '2xl': 'shadow-2xl dark:shadow-dark-2xl',
    };

    // Base classes
    const baseClasses = `
      transition-all duration-300
      ${fullWidth ? 'w-full' : ''}
    `;

    // Hover classes
    const hoverClasses = hoverable ? `
      hover:shadow-lg dark:hover:shadow-dark-lg
      hover:-translate-y-1
    ` : '';

    // Clickable classes
    const clickableClasses = clickable ? `
      cursor-pointer
      active:scale-[0.98]
    ` : '';

    // Combine all classes
    const cardClasses = cn(
      baseClasses,
      variants[variant],
      variant === 'outlined' && borderColors[borderColor],
      variant === 'filled' && bgColors[bgColor],
      paddings[padding],
      roundedStyles[rounded],
      variant === 'elevated' && shadowStyles[shadow],
      hoverClasses,
      clickableClasses,
      className
    );

    return (
      <motion.div
        ref={ref}
        className={cardClasses}
        whileHover={hoverable ? { y: -4 } : undefined}
        whileTap={clickable ? { scale: 0.98 } : undefined}
        transition={{ duration: 0.2 }}
        {...props}
      >
        {children}
      </motion.div>
    );
  }
);

Card.displayName = 'Card';

/**
 * Card Header Component
 */
export interface CardHeaderProps extends HTMLAttributes<HTMLDivElement> {
  /**
   * Title text
   */
  title?: string;

  /**
   * Subtitle text
   */
  subtitle?: string;

  /**
   * Action element (button, menu, etc)
   */
  action?: React.ReactNode;

  /**
   * Border bottom
   */
  bordered?: boolean;

  /**
   * Additional class names
   */
  className?: string;

  /**
   * Children elements
   */
  children?: React.ReactNode;
}

export const CardHeader: React.FC<CardHeaderProps> = ({
  title,
  subtitle,
  action,
  bordered = false,
  className,
  children,
  ...props
}) => {
  const headerClasses = cn(
    'px-6 py-4',
    bordered && 'border-b border-gray-200 dark:border-gray-700',
    className
  );

  if (children) {
    return (
      <div className={headerClasses} {...props}>
        {children}
      </div>
    );
  }

  return (
    <div className={headerClasses} {...props}>
      <div className="flex items-start justify-between">
        <div>
          {title && (
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {title}
            </h3>
          )}
          {subtitle && (
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              {subtitle}
            </p>
          )}
        </div>
        {action && <div>{action}</div>}
      </div>
    </div>
  );
};

CardHeader.displayName = 'CardHeader';

/**
 * Card Body Component
 */
export interface CardBodyProps extends HTMLAttributes<HTMLDivElement> {
  /**
   * Padding size
   */
  padding?: 'none' | 'sm' | 'md' | 'lg';

  /**
   * Additional class names
   */
  className?: string;

  /**
   * Children elements
   */
  children?: React.ReactNode;
}

export const CardBody: React.FC<CardBodyProps> = ({
  padding = 'md',
  className,
  children,
  ...props
}) => {
  const paddings = {
    none: '',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  };

  const bodyClasses = cn(
    paddings[padding],
    className
  );

  return (
    <div className={bodyClasses} {...props}>
      {children}
    </div>
  );
};

CardBody.displayName = 'CardBody';

/**
 * Card Footer Component
 */
export interface CardFooterProps extends HTMLAttributes<HTMLDivElement> {
  /**
   * Border top
   */
  bordered?: boolean;

  /**
   * Alignment of footer content
   */
  align?: 'left' | 'center' | 'right' | 'between';

  /**
   * Additional class names
   */
  className?: string;

  /**
   * Children elements
   */
  children?: React.ReactNode;
}

export const CardFooter: React.FC<CardFooterProps> = ({
  bordered = false,
  align = 'right',
  className,
  children,
  ...props
}) => {
  const alignments = {
    left: 'justify-start',
    center: 'justify-center',
    right: 'justify-end',
    between: 'justify-between',
  };

  const footerClasses = cn(
    'px-6 py-4 flex items-center',
    alignments[align],
    bordered && 'border-t border-gray-200 dark:border-gray-700',
    className
  );

  return (
    <div className={footerClasses} {...props}>
      {children}
    </div>
  );
};

CardFooter.displayName = 'CardFooter';

export default Card;