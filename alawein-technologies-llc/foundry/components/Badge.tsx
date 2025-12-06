/**
 * Badge Component
 * A small labeling component for highlighting status or attributes
 */

import React from 'react';
import { motion } from 'framer-motion';
import { X } from 'lucide-react';
import { cn } from '../utils/cn';

export interface BadgeProps {
  /**
   * Badge variant
   */
  variant?: 'default' | 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'info';

  /**
   * Badge size
   */
  size?: 'xs' | 'sm' | 'md' | 'lg';

  /**
   * Badge style
   */
  style?: 'solid' | 'subtle' | 'outline';

  /**
   * Rounded corners
   */
  rounded?: 'sm' | 'md' | 'lg' | 'full';

  /**
   * Removable badge
   */
  removable?: boolean;

  /**
   * On remove handler
   */
  onRemove?: () => void;

  /**
   * Dot indicator
   */
  dot?: boolean;

  /**
   * Dot color (for custom dot colors)
   */
  dotColor?: string;

  /**
   * Icon to display before text
   */
  icon?: React.ReactNode;

  /**
   * Animate on appear
   */
  animate?: boolean;

  /**
   * Additional class names
   */
  className?: string;

  /**
   * Children content
   */
  children: React.ReactNode;
}

const Badge: React.FC<BadgeProps> = ({
  variant = 'default',
  size = 'md',
  style = 'subtle',
  rounded = 'md',
  removable = false,
  onRemove,
  dot = false,
  dotColor,
  icon,
  animate = false,
  className,
  children,
}) => {
  // Size styles
  const sizes = {
    xs: 'px-1.5 py-0.5 text-xs',
    sm: 'px-2 py-0.5 text-sm',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base',
  };

  // Rounded styles
  const roundedStyles = {
    sm: 'rounded-sm',
    md: 'rounded-md',
    lg: 'rounded-lg',
    full: 'rounded-full',
  };

  // Variant styles based on style type
  const getVariantStyles = () => {
    const variantMap = {
      default: {
        solid: 'bg-gray-500 text-white',
        subtle: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300',
        outline: 'border border-gray-300 text-gray-700 dark:border-gray-600 dark:text-gray-300',
      },
      primary: {
        solid: 'bg-primary-500 text-white',
        subtle: 'bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300',
        outline: 'border border-primary-300 text-primary-700 dark:border-primary-700 dark:text-primary-300',
      },
      secondary: {
        solid: 'bg-secondary-500 text-white',
        subtle: 'bg-secondary-100 text-secondary-700 dark:bg-secondary-900/30 dark:text-secondary-300',
        outline: 'border border-secondary-300 text-secondary-700 dark:border-secondary-700 dark:text-secondary-300',
      },
      success: {
        solid: 'bg-success-500 text-white',
        subtle: 'bg-success-100 text-success-700 dark:bg-success-900/30 dark:text-success-300',
        outline: 'border border-success-300 text-success-700 dark:border-success-700 dark:text-success-300',
      },
      warning: {
        solid: 'bg-warning-500 text-white',
        subtle: 'bg-warning-100 text-warning-700 dark:bg-warning-900/30 dark:text-warning-300',
        outline: 'border border-warning-300 text-warning-700 dark:border-warning-700 dark:text-warning-300',
      },
      danger: {
        solid: 'bg-danger-500 text-white',
        subtle: 'bg-danger-100 text-danger-700 dark:bg-danger-900/30 dark:text-danger-300',
        outline: 'border border-danger-300 text-danger-700 dark:border-danger-700 dark:text-danger-300',
      },
      info: {
        solid: 'bg-info-500 text-white',
        subtle: 'bg-info-100 text-info-700 dark:bg-info-900/30 dark:text-info-300',
        outline: 'border border-info-300 text-info-700 dark:border-info-700 dark:text-info-300',
      },
    };

    return variantMap[variant]?.[style] || variantMap.default.subtle;
  };

  // Base classes
  const baseClasses = 'inline-flex items-center gap-1.5 font-medium transition-all';

  // Combine classes
  const badgeClasses = cn(
    baseClasses,
    sizes[size],
    roundedStyles[rounded],
    getVariantStyles(),
    className
  );

  const content = (
    <>
      {/* Dot indicator */}
      {dot && (
        <span
          className={cn(
            'inline-block rounded-full',
            size === 'xs' && 'w-1.5 h-1.5',
            size === 'sm' && 'w-2 h-2',
            size === 'md' && 'w-2 h-2',
            size === 'lg' && 'w-2.5 h-2.5'
          )}
          style={{ backgroundColor: dotColor || 'currentColor' }}
        />
      )}

      {/* Icon */}
      {icon && (
        <span className="inline-flex items-center">
          {icon}
        </span>
      )}

      {/* Content */}
      <span>{children}</span>

      {/* Remove button */}
      {removable && (
        <button
          onClick={onRemove}
          className="ml-1 -mr-0.5 inline-flex items-center justify-center rounded-full hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
        >
          <X className={cn(
            size === 'xs' && 'w-2.5 h-2.5',
            size === 'sm' && 'w-3 h-3',
            size === 'md' && 'w-3.5 h-3.5',
            size === 'lg' && 'w-4 h-4'
          )} />
        </button>
      )}
    </>
  );

  if (animate) {
    return (
      <motion.span
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0, opacity: 0 }}
        transition={{ duration: 0.2, type: 'spring', stiffness: 300 }}
        className={badgeClasses}
      >
        {content}
      </motion.span>
    );
  }

  return (
    <span className={badgeClasses}>
      {content}
    </span>
  );
};

Badge.displayName = 'Badge';

export default Badge;