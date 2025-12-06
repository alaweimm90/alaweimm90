import React from 'react';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Atom, Loader2, Zap } from 'lucide-react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  message?: string;
  className?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  size = 'md', 
  message,
  className = '' 
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  };

  return (
    <div className={`flex items-center justify-center gap-3 ${className}`}>
      <Loader2 className={`${sizeClasses[size]} animate-spin text-primary`} />
      {message && (
        <span className="text-sm text-muted-foreground">{message}</span>
      )}
    </div>
  );
};

interface QuantumLoadingProps {
  message?: string;
  intensity?: number;
}

export const QuantumLoading: React.FC<QuantumLoadingProps> = ({ 
  message = 'Loading...',
  intensity = 1 
}) => {
  return (
    <div className="flex flex-col items-center justify-center gap-6 p-8">
      <div className="relative">
        {/* Quantum particle animation */}
        <div className="relative w-16 h-16">
          <div 
            className="absolute inset-0 border-2 border-primary/30 rounded-full animate-spin"
            style={{ animationDuration: `${2 / intensity}s` }}
          />
          <div 
            className="absolute inset-2 border-2 border-accent/50 rounded-full animate-spin"
            style={{ 
              animationDuration: `${1.5 / intensity}s`,
              animationDirection: 'reverse'
            }}
          />
          <div 
            className="absolute inset-4 border-2 border-primary-glow/70 rounded-full animate-spin"
            style={{ animationDuration: `${1 / intensity}s` }}
          />
          <div className="absolute inset-6 bg-gradient-quantum rounded-full animate-pulse" />
          <Atom className="absolute inset-0 m-auto w-6 h-6 text-primary-foreground animate-pulse" />
        </div>
      </div>
      
      <div className="text-center">
        <p className="text-lg font-medium text-foreground mb-2">{message}</p>
        <div className="flex items-center justify-center gap-1">
          {Array.from({ length: 3 }).map((_, i) => (
            <div
              key={i}
              className="w-2 h-2 bg-primary rounded-full animate-pulse"
              style={{ 
                animationDelay: `${i * 0.2}s`,
                animationDuration: '1s'
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export const ModuleCardLoader: React.FC = () => {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-6 w-16 rounded-full" />
        </div>
        <Skeleton className="h-6 w-3/4" />
        <Skeleton className="h-4 w-full" />
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Skeleton className="h-5 w-16" />
          <Skeleton className="h-5 w-20" />
          <Skeleton className="h-5 w-14" />
        </div>
        <Skeleton className="h-16 w-full rounded-lg" />
        <div className="flex gap-2">
          <Skeleton className="h-10 flex-1" />
          <Skeleton className="h-10 w-20" />
        </div>
      </CardContent>
    </Card>
  );
};

export const PhysicsPlotLoader: React.FC<{ height?: number }> = ({ height = 400 }) => {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-8 w-24" />
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex justify-between items-end">
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-4 w-16" />
            <Skeleton className="h-4 w-16" />
          </div>
          
          <Skeleton className={`w-full`} style={{ height: `${height}px` }} />
          
          <div className="flex justify-center">
            <Skeleton className="h-4 w-24" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

interface DataLoadingProps {
  type: 'calculation' | 'simulation' | 'rendering' | 'analysis';
  progress?: number;
  message?: string;
}

export const DataLoading: React.FC<DataLoadingProps> = ({ 
  type,
  progress,
  message 
}) => {
  const typeConfig = {
    calculation: {
      icon: Zap,
      color: 'text-primary',
      defaultMessage: 'Performing calculations...'
    },
    simulation: {
      icon: Atom,
      color: 'text-accent',
      defaultMessage: 'Running simulation...'
    },
    rendering: {
      icon: Loader2,
      color: 'text-primary-glow',
      defaultMessage: 'Rendering visualization...'
    },
    analysis: {
      icon: Zap,
      color: 'text-primary',
      defaultMessage: 'Analyzing results...'
    }
  };

  const config = typeConfig[type];
  const Icon = config.icon;

  return (
    <div className="flex flex-col items-center justify-center gap-4 p-6">
      <div className="relative">
        <div className="w-12 h-12 bg-muted rounded-full animate-pulse flex items-center justify-center">
          <Icon className={`w-6 h-6 ${config.color} ${Icon === Loader2 ? 'animate-spin' : 'animate-pulse'}`} />
        </div>
        
        {progress !== undefined && (
          <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2">
            <div className="bg-background px-2 py-1 rounded text-xs font-medium text-muted-foreground">
              {Math.round(progress)}%
            </div>
          </div>
        )}
      </div>
      
      <div className="text-center">
        <p className="text-sm font-medium text-foreground">
          {message || config.defaultMessage}
        </p>
        
        {progress !== undefined && (
          <div className="w-48 h-2 bg-muted rounded-full mt-2 overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-300 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export const TableLoader: React.FC<{ rows?: number; columns?: number }> = ({ 
  rows = 5, 
  columns = 4 
}) => {
  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex gap-4 pb-2 border-b">
        {Array.from({ length: columns }).map((_, i) => (
          <Skeleton key={i} className="h-5 flex-1" />
        ))}
      </div>
      
      {/* Rows */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <div key={rowIndex} className="flex gap-4 py-2">
          {Array.from({ length: columns }).map((_, colIndex) => (
            <Skeleton 
              key={colIndex} 
              className={`h-4 flex-1 ${Math.random() > 0.7 ? 'w-3/4' : ''}`}
            />
          ))}
        </div>
      ))}
    </div>
  );
};

interface PageLoadingProps {
  title?: string;
  description?: string;
}

export const PageLoading: React.FC<PageLoadingProps> = ({ 
  title = 'Loading Module',
  description = 'Preparing physics simulation...'
}) => {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="absolute inset-0 bg-gradient-cosmic" />
      <Card className="w-full max-w-md mx-4">
        <CardHeader className="text-center">
          <QuantumLoading message={title} />
        </CardHeader>
        <CardContent className="text-center">
          <p className="text-muted-foreground text-sm">{description}</p>
        </CardContent>
      </Card>
    </div>
  );
};

export default {
  LoadingSpinner,
  QuantumLoading,
  ModuleCardLoader,
  PhysicsPlotLoader,
  DataLoading,
  TableLoader,
  PageLoading,
};