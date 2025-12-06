/**
 * Spinner Component
 * Loading spinner with various styles and sizes
 */

import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '../utils/cn';

export interface SpinnerProps {
  /**
   * Spinner size
   */
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';

  /**
   * Spinner variant
   */
  variant?: 'circle' | 'dots' | 'pulse' | 'wave' | 'bounce';

  /**
   * Spinner color
   */
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'current';

  /**
   * Loading text
   */
  label?: string;

  /**
   * Full screen overlay
   */
  fullScreen?: boolean;

  /**
   * Center in container
   */
  center?: boolean;

  /**
   * Additional class names
   */
  className?: string;
}

const Spinner: React.FC<SpinnerProps> = ({
  size = 'md',
  variant = 'circle',
  color = 'primary',
  label,
  fullScreen = false,
  center = false,
  className,
}) => {
  // Size dimensions
  const sizes = {
    xs: 16,
    sm: 20,
    md: 24,
    lg: 32,
    xl: 48,
  };

  // Color styles
  const colors = {
    primary: 'text-primary-500',
    secondary: 'text-secondary-500',
    success: 'text-success-500',
    warning: 'text-warning-500',
    danger: 'text-danger-500',
    current: 'text-current',
  };

  const dimension = sizes[size];
  const colorClass = colors[color];

  // Circle spinner
  const CircleSpinner = () => (
    <svg
      className={cn('animate-spin', colorClass)}
      width={dimension}
      height={dimension}
      viewBox="0 0 24 24"
      fill="none"
    >
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
  );

  // Dots spinner
  const DotsSpinner = () => (
    <div className={cn('flex gap-1', colorClass)}>
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="rounded-full bg-current"
          style={{
            width: dimension / 3,
            height: dimension / 3,
          }}
          animate={{
            scale: [1, 1.5, 1],
            opacity: [1, 0.5, 1],
          }}
          transition={{
            duration: 0.6,
            repeat: Infinity,
            delay: i * 0.2,
          }}
        />
      ))}
    </div>
  );

  // Pulse spinner
  const PulseSpinner = () => (
    <div className={cn('relative', colorClass)}>
      <motion.div
        className="absolute inset-0 rounded-full bg-current"
        animate={{
          scale: [1, 1.5],
          opacity: [0.5, 0],
        }}
        transition={{
          duration: 1,
          repeat: Infinity,
        }}
      />
      <div
        className="rounded-full bg-current opacity-50"
        style={{ width: dimension, height: dimension }}
      />
    </div>
  );

  // Wave spinner
  const WaveSpinner = () => (
    <div className={cn('flex gap-1', colorClass)}>
      {[0, 1, 2, 3, 4].map((i) => (
        <motion.div
          key={i}
          className="bg-current"
          style={{
            width: dimension / 6,
            height: dimension,
          }}
          animate={{
            scaleY: [0.4, 1, 0.4],
          }}
          transition={{
            duration: 0.8,
            repeat: Infinity,
            delay: i * 0.1,
          }}
        />
      ))}
    </div>
  );

  // Bounce spinner
  const BounceSpinner = () => (
    <div className={cn('flex gap-1', colorClass)}>
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="rounded-full bg-current"
          style={{
            width: dimension / 3,
            height: dimension / 3,
          }}
          animate={{
            y: [0, -dimension / 2, 0],
          }}
          transition={{
            duration: 0.6,
            repeat: Infinity,
            delay: i * 0.15,
          }}
        />
      ))}
    </div>
  );

  // Render appropriate spinner variant
  const renderSpinner = () => {
    switch (variant) {
      case 'dots':
        return <DotsSpinner />;
      case 'pulse':
        return <PulseSpinner />;
      case 'wave':
        return <WaveSpinner />;
      case 'bounce':
        return <BounceSpinner />;
      case 'circle':
      default:
        return <CircleSpinner />;
    }
  };

  // Wrapper classes
  const wrapperClasses = cn(
    'inline-flex flex-col items-center justify-center gap-2',
    center && 'absolute inset-0 m-auto',
    fullScreen && 'fixed inset-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm',
    className
  );

  return (
    <div className={wrapperClasses}>
      {renderSpinner()}
      {label && (
        <span className={cn(
          'text-gray-600 dark:text-gray-400',
          size === 'xs' && 'text-xs',
          size === 'sm' && 'text-sm',
          size === 'md' && 'text-base',
          size === 'lg' && 'text-lg',
          size === 'xl' && 'text-xl'
        )}>
          {label}
        </span>
      )}
    </div>
  );
};

Spinner.displayName = 'Spinner';

/**
 * Skeleton Loading Component
 * Placeholder loading state for content
 */
export interface SkeletonProps {
  /**
   * Skeleton variant
   */
  variant?: 'text' | 'circular' | 'rectangular' | 'rounded';

  /**
   * Width (CSS value)
   */
  width?: string | number;

  /**
   * Height (CSS value)
   */
  height?: string | number;

  /**
   * Animation type
   */
  animation?: 'pulse' | 'wave' | 'none';

  /**
   * Number of lines (for text variant)
   */
  lines?: number;

  /**
   * Additional class names
   */
  className?: string;
}

export const Skeleton: React.FC<SkeletonProps> = ({
  variant = 'text',
  width,
  height,
  animation = 'pulse',
  lines = 1,
  className,
}) => {
  // Base styles
  const baseClasses = 'bg-gray-200 dark:bg-gray-700';

  // Variant styles
  const variantStyles = {
    text: 'h-4 rounded',
    circular: 'rounded-full',
    rectangular: '',
    rounded: 'rounded-lg',
  };

  // Animation styles
  const animationStyles = {
    pulse: 'animate-pulse',
    wave: 'relative overflow-hidden',
    none: '',
  };

  // Wave animation overlay
  const WaveOverlay = () => (
    <motion.div
      className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/20 to-transparent"
      animate={{ x: '200%' }}
      transition={{ repeat: Infinity, duration: 1.5, ease: 'linear' }}
    />
  );

  const skeletonClasses = cn(
    baseClasses,
    variantStyles[variant],
    animationStyles[animation],
    className
  );

  // Render multiple lines for text variant
  if (variant === 'text' && lines > 1) {
    return (
      <div className="space-y-2">
        {Array.from({ length: lines }, (_, i) => (
          <div
            key={i}
            className={skeletonClasses}
            style={{
              width: width || (i === lines - 1 ? '80%' : '100%'),
              height: height || undefined,
            }}
          >
            {animation === 'wave' && <WaveOverlay />}
          </div>
        ))}
      </div>
    );
  }

  return (
    <div
      className={skeletonClasses}
      style={{
        width: width || (variant === 'circular' ? 40 : '100%'),
        height: height || (variant === 'circular' ? 40 : variant === 'text' ? 16 : 120),
      }}
    >
      {animation === 'wave' && <WaveOverlay />}
    </div>
  );
};

Skeleton.displayName = 'Skeleton';

export default Spinner;