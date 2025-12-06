import React from 'react';
import { Card, CardContent, CardHeader } from '@/ui/molecules/Card';
import { Skeleton } from '@/ui/atoms/Skeleton';
import { cn } from '@/lib/utils';

// Base skeleton component
const SkeletonBase: React.FC<{ className?: string; style?: React.CSSProperties }> = ({ className, style }) => (
  <div className={cn('animate-pulse bg-gray-200 rounded', className)} style={style} />
);

// Loading spinner with message
export const LoadingSpinner: React.FC<{ 
  size?: 'sm' | 'md' | 'lg';
  message?: string;
  className?: string;
}> = ({ size = 'md', message, className }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8', 
    lg: 'w-12 h-12'
  };

  return (
    <div className={cn('flex flex-col items-center justify-center gap-3', className)}>
      <div className={cn(
        'border-2 border-gray-300 border-t-repz-primary rounded-full animate-spin',
        sizeClasses[size]
      )} />
      {message && (
        <p className="text-sm text-gray-600 text-center">{message}</p>
      )}
    </div>
  );
};

// Dashboard loading skeleton
export const DashboardSkeleton: React.FC = () => (
  <div className="space-y-6">
    {/* Header skeleton */}
    <div className="flex items-center justify-between">
      <SkeletonBase className="h-8 w-48" />
      <SkeletonBase className="h-10 w-32" />
    </div>
    
    {/* Stats cards skeleton */}
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <Card key={i}>
          <CardContent className="p-6">
            <div className="space-y-2">
              <SkeletonBase className="h-4 w-20" />
              <SkeletonBase className="h-8 w-16" />
              <SkeletonBase className="h-3 w-24" />
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
    
    {/* Main content skeleton */}
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardHeader>
          <SkeletonBase className="h-6 w-32" />
          <SkeletonBase className="h-4 w-48" />
        </CardHeader>
        <CardContent>
          <SkeletonBase className="h-64 w-full" />
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <SkeletonBase className="h-6 w-32" />
        </CardHeader>
        <CardContent className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="flex items-center gap-3">
              <SkeletonBase className="h-10 w-10 rounded-full" />
              <div className="flex-1 space-y-1">
                <SkeletonBase className="h-4 w-full" />
                <SkeletonBase className="h-3 w-2/3" />
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  </div>
);

// Table loading skeleton
export const TableSkeleton: React.FC<{ rows?: number; columns?: number }> = ({ 
  rows = 5, 
  columns = 4 
}) => (
  <div className="space-y-3">
    {/* Table header */}
    <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
      {Array.from({ length: columns }).map((_, i) => (
        <SkeletonBase key={i} className="h-4 w-full" />
      ))}
    </div>
    
    {/* Table rows */}
    {Array.from({ length: rows }).map((_, rowIndex) => (
      <div 
        key={rowIndex} 
        className="grid gap-4" 
        style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}
      >
        {Array.from({ length: columns }).map((_, colIndex) => (
          <SkeletonBase key={colIndex} className="h-6 w-full" />
        ))}
      </div>
    ))}
  </div>
);

// Card list loading skeleton
export const CardListSkeleton: React.FC<{ count?: number }> = ({ count = 3 }) => (
  <div className="space-y-4">
    {Array.from({ length: count }).map((_, i) => (
      <Card key={i}>
        <CardHeader>
          <div className="flex items-center gap-3">
            <SkeletonBase className="h-12 w-12 rounded-full" />
            <div className="space-y-2">
              <SkeletonBase className="h-5 w-32" />
              <SkeletonBase className="h-4 w-48" />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <SkeletonBase className="h-4 w-full" />
            <SkeletonBase className="h-4 w-5/6" />
            <SkeletonBase className="h-4 w-3/4" />
          </div>
        </CardContent>
      </Card>
    ))}
  </div>
);

// Chart loading skeleton
export const ChartSkeleton: React.FC<{ height?: number }> = ({ height = 300 }) => (
  <Card>
    <CardHeader>
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <SkeletonBase className="h-6 w-32" />
          <SkeletonBase className="h-4 w-48" />
        </div>
        <SkeletonBase className="h-8 w-24" />
      </div>
    </CardHeader>
    <CardContent>
      <div className="space-y-4">
        {/* Chart legend */}
        <div className="flex gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="flex items-center gap-2">
              <SkeletonBase className="h-3 w-3 rounded-full" />
              <SkeletonBase className="h-3 w-16" />
            </div>
          ))}
        </div>
        
        {/* Chart area */}
        <SkeletonBase className="w-full" style={{ height: `${height}px` }} />
      </div>
    </CardContent>
  </Card>
);

// Form loading skeleton
export const FormSkeleton: React.FC<{ fields?: number }> = ({ fields = 5 }) => (
  <Card>
    <CardHeader>
      <SkeletonBase className="h-6 w-48" />
      <SkeletonBase className="h-4 w-64" />
    </CardHeader>
    <CardContent className="space-y-6">
      {Array.from({ length: fields }).map((_, i) => (
        <div key={i} className="space-y-2">
          <SkeletonBase className="h-4 w-24" />
          <SkeletonBase className="h-10 w-full" />
        </div>
      ))}
      
      <div className="flex gap-3 pt-4">
        <SkeletonBase className="h-10 w-24" />
        <SkeletonBase className="h-10 w-20" />
      </div>
    </CardContent>
  </Card>
);

// Profile loading skeleton
export const ProfileSkeleton: React.FC = () => (
  <Card>
    <CardContent className="pt-6">
      <div className="flex flex-col items-center space-y-4">
        <SkeletonBase className="h-24 w-24 rounded-full" />
        <div className="text-center space-y-2">
          <SkeletonBase className="h-6 w-32" />
          <SkeletonBase className="h-4 w-48" />
        </div>
        
        <div className="w-full space-y-3 pt-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex justify-between">
              <SkeletonBase className="h-4 w-20" />
              <SkeletonBase className="h-4 w-32" />
            </div>
          ))}
        </div>
      </div>
    </CardContent>
  </Card>
);

// Generic content skeleton with customizable layout
export const ContentSkeleton: React.FC<{
  lines?: number;
  showHeader?: boolean;
  showImage?: boolean;
  className?: string;
}> = ({ lines = 3, showHeader = true, showImage = false, className }) => (
  <div className={cn('space-y-4', className)}>
    {showHeader && (
      <div className="space-y-2">
        <SkeletonBase className="h-6 w-2/3" />
        <SkeletonBase className="h-4 w-1/2" />
      </div>
    )}
    
    {showImage && (
      <SkeletonBase className="h-48 w-full rounded-lg" />
    )}
    
    <div className="space-y-2">
      {Array.from({ length: lines }).map((_, i) => (
        <SkeletonBase 
          key={i} 
          className={cn(
            'h-4',
            i === lines - 1 ? 'w-3/4' : 'w-full'
          )} 
        />
      ))}
    </div>
  </div>
);