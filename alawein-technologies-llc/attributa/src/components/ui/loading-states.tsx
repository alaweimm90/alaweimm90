import React from 'react';
import { cn } from '@/lib/utils';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  color?: 'primary' | 'accent' | 'muted';
}

export function LoadingSpinner({ size = 'md', className, color = 'primary' }: LoadingSpinnerProps) {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  };

  const colors = {
    primary: 'text-primary',
    accent: 'text-accent',
    muted: 'text-muted-foreground'
  };

  return (
    <div
      className={cn(
        'animate-spin rounded-full border-2 border-transparent border-t-current',
        sizes[size],
        colors[color],
        className
      )}
      role="status"
      aria-label="Loading"
    >
      <span className="sr-only">Loading...</span>
    </div>
  );
}

interface LoadingDotsProps {
  className?: string;
  color?: 'primary' | 'accent' | 'muted';
}

export function LoadingDots({ className, color = 'primary' }: LoadingDotsProps) {
  const colors = {
    primary: 'bg-primary',
    accent: 'bg-accent',
    muted: 'bg-muted-foreground'
  };

  return (
    <div className={cn('flex space-x-1', className)} role="status" aria-label="Loading">
      <div className={cn('w-2 h-2 rounded-full animate-bounce', colors[color])} style={{ animationDelay: '0ms' }} />
      <div className={cn('w-2 h-2 rounded-full animate-bounce', colors[color])} style={{ animationDelay: '150ms' }} />
      <div className={cn('w-2 h-2 rounded-full animate-bounce', colors[color])} style={{ animationDelay: '300ms' }} />
      <span className="sr-only">Loading...</span>
    </div>
  );
}

interface LoadingPulseProps {
  className?: string;
}

export function LoadingPulse({ className }: LoadingPulseProps) {
  return (
    <div className={cn('flex space-x-2', className)} role="status" aria-label="Loading">
      <div className="w-3 h-3 bg-primary rounded-full animate-pulse" />
      <div className="w-3 h-3 bg-accent rounded-full animate-pulse" style={{ animationDelay: '0.2s' }} />
      <div className="w-3 h-3 bg-secondary rounded-full animate-pulse" style={{ animationDelay: '0.4s' }} />
      <span className="sr-only">Loading...</span>
    </div>
  );
}