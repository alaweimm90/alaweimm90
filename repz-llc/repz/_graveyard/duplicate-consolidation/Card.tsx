import React from 'react';
import { cn } from '@/lib/utils';

/**
 * Atomic Design - Card Molecule
 * Combines multiple atoms into a cohesive interface element
 * All styling derived from design tokens
 */

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'elevated' | 'outlined' | 'glass';
  padding?: 'sm' | 'md' | 'lg' | 'xl';
  interactive?: boolean;
  children: React.ReactNode;
}

export interface CardHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export interface CardContentProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export interface CardFooterProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(({
  className,
  variant = 'default',
  padding = 'md',
  interactive = false,
  children,
  ...props
}, ref) => {
  return (
    <div
      className={cn(
        'repz-card',
        'repz-card--' + variant,
        'repz-card--padding-' + padding,
        interactive && 'repz-card--interactive',
        className
      )}
      ref={ref}
      {...props}
    >
      {children}
    </div>
  );
});

export const CardHeader = React.forwardRef<HTMLDivElement, CardHeaderProps>(({
  className,
  children,
  ...props
}, ref) => {
  return (
    <div
      className={cn('repz-card__header', className)}
      ref={ref}
      {...props}
    >
      {children}
    </div>
  );
});

export const CardContent = React.forwardRef<HTMLDivElement, CardContentProps>(({
  className,
  children,
  ...props
}, ref) => {
  return (
    <div
      className={cn('repz-card__content', className)}
      ref={ref}
      {...props}
    >
      {children}
    </div>
  );
});

export const CardFooter = React.forwardRef<HTMLDivElement, CardFooterProps>(({
  className,
  children,
  ...props
}, ref) => {
  return (
    <div
      className={cn('repz-card__footer', className)}
      ref={ref}
      {...props}
    >
      {children}
    </div>
  );
});

Card.displayName = 'Card';
CardHeader.displayName = 'CardHeader';
CardContent.displayName = 'CardContent';
CardFooter.displayName = 'CardFooter';