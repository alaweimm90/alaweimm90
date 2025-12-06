import React from 'react';
import { Skeleton } from '@/ui/atoms/Skeleton';
import { Card, CardContent, CardHeader } from '@/ui/molecules/Card';

export const DashboardSkeleton = () => (
  <div className="space-y-6">
    {/* Header Skeleton */}
    <div className="flex justify-between items-center">
      <div className="space-y-2">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-4 w-32" />
      </div>
      <Skeleton className="h-10 w-32" />
    </div>

    {/* Metrics Cards Skeleton */}
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {[...Array(4)].map((_, i) => (
        <Card key={i}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-4 w-4" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-8 w-16 mb-2" />
            <Skeleton className="h-3 w-32" />
          </CardContent>
        </Card>
      ))}
    </div>

    {/* Chart Skeleton */}
    <Card>
      <CardHeader>
        <Skeleton className="h-6 w-32" />
      </CardHeader>
      <CardContent>
        <Skeleton className="h-64 w-full" />
      </CardContent>
    </Card>
  </div>
);

export const TableSkeleton = ({ rows = 5, columns = 4 }: { rows?: number; columns?: number }) => (
  <div className="space-y-4">
    <div className="flex justify-between items-center">
      <Skeleton className="h-6 w-48" />
      <Skeleton className="h-10 w-32" />
    </div>
    
    <div className="border rounded-lg">
      {/* Header */}
      <div className="grid gap-4 p-4 border-b" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
        {[...Array(columns)].map((_, i) => (
          <Skeleton key={i} className="h-4 w-24" />
        ))}
      </div>
      
      {/* Rows */}
      {[...Array(rows)].map((_, rowIndex) => (
        <div key={rowIndex} className="grid gap-4 p-4 border-b last:border-b-0" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
          {[...Array(columns)].map((_, colIndex) => (
            <Skeleton key={colIndex} className="h-4 w-full" />
          ))}
        </div>
      ))}
    </div>
  </div>
);

export const ProfileSkeleton = () => (
  <Card>
    <CardHeader>
      <div className="flex items-center space-x-4">
        <Skeleton className="h-16 w-16 rounded-full" />
        <div className="space-y-2">
          <Skeleton className="h-6 w-32" />
          <Skeleton className="h-4 w-24" />
        </div>
      </div>
    </CardHeader>
    <CardContent className="space-y-4">
      {[...Array(6)].map((_, i) => (
        <div key={i} className="grid grid-cols-2 gap-4">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-32" />
        </div>
      ))}
    </CardContent>
  </Card>
);

export const WorkoutSkeleton = () => (
  <div className="space-y-4">
    {[...Array(3)].map((_, i) => (
      <Card key={i}>
        <CardHeader>
          <div className="flex justify-between items-start">
            <div className="space-y-2">
              <Skeleton className="h-5 w-48" />
              <Skeleton className="h-4 w-32" />
            </div>
            <Skeleton className="h-6 w-16" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(4)].map((_, j) => (
              <div key={j} className="flex justify-between items-center">
                <div className="space-y-1">
                  <Skeleton className="h-4 w-32" />
                  <Skeleton className="h-3 w-24" />
                </div>
                <Skeleton className="h-8 w-20" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    ))}
  </div>
);

export const ChartSkeleton = ({ height = "h-64" }: { height?: string }) => (
  <Card>
    <CardHeader>
      <div className="flex justify-between items-center">
        <Skeleton className="h-6 w-32" />
        <Skeleton className="h-8 w-24" />
      </div>
    </CardHeader>
    <CardContent>
      <Skeleton className={`${height} w-full`} />
    </CardContent>
  </Card>
);