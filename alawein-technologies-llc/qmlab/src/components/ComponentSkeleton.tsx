import React from 'react';
import { Skeleton } from '@/components/ui/skeleton';

interface ComponentSkeletonProps {
  className?: string;
  error?: string;
  message?: string;
  showProgressBar?: boolean;
}

export const ComponentSkeleton: React.FC<ComponentSkeletonProps> = ({ className = '', error, message, showProgressBar }) => {
  if (error) {
    return (
      <div className={`rounded-3xl border border-red-400/30 bg-gradient-to-br from-red-500/10 via-slate-900/50 to-slate-900/30 backdrop-blur-sm shadow-2xl p-8 space-y-4 ${className}`}>
        <div className="flex items-center gap-3 text-red-400">
          <div className="w-8 h-8 rounded-lg bg-red-400/20 border border-red-400/30 flex items-center justify-center">
            <span className="text-lg">⚠</span>
          </div>
          <div>
            <h3 className="font-semibold text-red-300">Component Load Error</h3>
            <p className="text-sm text-red-400/80">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`rounded-3xl border border-blue-400/30 bg-gradient-to-br from-blue-500/10 via-slate-900/50 to-slate-900/30 backdrop-blur-sm shadow-2xl p-8 space-y-6 animate-quantum-pulse-glow ${className}`}>
      {/* Quantum loading indicator */}
      <div className="flex items-center gap-4 mb-6">
        <div className="relative">
          <div className="w-12 h-12 rounded-lg bg-blue-400/20 border border-blue-400/30 animate-quantum-float-gentle">
            <div className="absolute inset-2 border border-blue-400/50 rounded animate-quantum-orbit"></div>
            <div className="absolute inset-3 w-2 h-2 bg-blue-400 rounded-full animate-pulse mx-auto mt-2"></div>
          </div>
        </div>
        <div className="flex-1 space-y-3">
          <Skeleton className="w-40 h-6 bg-gradient-to-r from-blue-400/30 to-purple-400/30 animate-quantum-gradient-shift" />
          <Skeleton className="w-56 h-4 bg-gradient-to-r from-purple-400/20 to-indigo-400/20 animate-quantum-gradient-shift animate-delay-300" />
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
          <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse animate-delay-200"></div>
          <div className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse animate-delay-500"></div>
        </div>
      </div>
      
      {/* Content skeleton with quantum effects */}
      <div className="space-y-4">
        <Skeleton className="w-full h-10 rounded-xl bg-gradient-to-r from-blue-400/20 to-purple-400/20 animate-quantum-gradient-shift" />
        <Skeleton className="w-full h-40 rounded-2xl bg-gradient-to-br from-blue-500/15 via-purple-500/10 to-indigo-500/15 animate-quantum-border-dance" />
      </div>
      
      {/* Quantum metrics skeleton */}
      <div className="grid grid-cols-3 gap-6 pt-6 border-t border-blue-400/20">
        <div className="text-center space-y-3">
          <Skeleton className="w-10 h-8 mx-auto rounded-lg bg-blue-400/30 animate-quantum-pulse-glow" />
          <Skeleton className="w-16 h-4 mx-auto bg-blue-400/20" />
        </div>
        <div className="text-center space-y-3">
          <Skeleton className="w-10 h-8 mx-auto rounded-lg bg-purple-400/30 animate-quantum-pulse-glow animate-delay-200" />
          <Skeleton className="w-16 h-4 mx-auto bg-purple-400/20" />
        </div>
        <div className="text-center space-y-3">
          <Skeleton className="w-10 h-8 mx-auto rounded-lg bg-indigo-400/30 animate-quantum-pulse-glow animate-delay-500" />
          <Skeleton className="w-16 h-4 mx-auto bg-indigo-400/20" />
        </div>
      </div>
      
      {/* Loading message */}
      <div className="text-center pt-4">
        <p className="text-sm text-slate-400 animate-pulse">
          {message || "Initializing quantum components..."}
        </p>
        {showProgressBar && (
          <div className="w-full bg-slate-700 rounded-full h-2 mt-2">
            <div className="bg-blue-400 h-2 rounded-full animate-pulse" style={{ width: '60%' }} />
          </div>
        )}
      </div>
    </div>
  );
};