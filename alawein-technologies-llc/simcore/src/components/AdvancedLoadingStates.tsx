import React, { useState, useEffect } from 'react';
import { useResponsiveEnhanced } from '@/hooks/use-responsive-enhanced';
import { cn } from '@/lib/utils';
import { Loader2, Zap, Cpu, Activity } from 'lucide-react';

// Skeleton loader for module cards
export const ModuleCardSkeleton = ({ className }: { className?: string }) => {
  return (
    <div className={cn(
      'rounded-lg border border-border bg-card p-4 sm:p-6 min-h-[120px] sm:min-h-[140px]',
      className
    )}>
      <div className="flex items-center gap-3 mb-3">
        <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-muted animate-pulse" />
        <div className="h-4 sm:h-5 bg-muted rounded flex-1 animate-pulse" />
      </div>
      <div className="space-y-2 mb-4">
        <div className="h-3 bg-muted rounded w-full animate-pulse" />
        <div className="h-3 bg-muted rounded w-3/4 animate-pulse" />
      </div>
      <div className="flex items-center justify-between">
        <div className="h-6 w-16 bg-muted rounded animate-pulse" />
        <div className="w-6 h-6 rounded-full bg-muted animate-pulse" />
      </div>
    </div>
  );
};

// Progressive loading indicator with stages
export const ProgressiveLoader = ({ 
  stage, 
  progress = 0,
  stages = ['Initializing', 'Loading Physics', 'Rendering', 'Ready']
}: { 
  stage: number;
  progress?: number;
  stages?: string[];
}) => {
  const { isMobile } = useResponsiveEnhanced();

  return (
    <div className="flex flex-col items-center gap-4 p-6">
      <div className="relative">
        <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-full border-4 border-muted">
          <div 
            className="absolute inset-0 rounded-full border-4 border-transparent border-t-primary transition-all duration-300"
            style={{ 
              transform: `rotate(${progress * 3.6}deg)`,
              borderTopColor: 'hsl(var(--primary))'
            }}
          />
        </div>
        <div className="absolute inset-0 flex items-center justify-center">
          {stage === 0 && <Loader2 className="w-6 h-6 sm:w-8 sm:h-8 text-primary animate-spin" />}
          {stage === 1 && <Cpu className="w-6 h-6 sm:w-8 sm:h-8 text-primary animate-pulse" />}
          {stage === 2 && <Activity className="w-6 h-6 sm:w-8 sm:h-8 text-primary animate-bounce" />}
          {stage === 3 && <Zap className="w-6 h-6 sm:w-8 sm:h-8 text-primary" />}
        </div>
      </div>
      
      <div className="text-center space-y-2">
        <p className="text-sm sm:text-base font-medium text-foreground">
          {stages[stage] || 'Loading...'}
        </p>
        <p className="text-xs sm:text-sm text-muted-foreground">
          {Math.round(progress)}% complete
        </p>
      </div>
      
      <div className="w-full max-w-xs">
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <div 
            className="h-full bg-primary transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <div className="flex gap-2 mt-2">
        {stages.map((_, index) => (
          <div
            key={index}
            className={cn(
              'w-2 h-2 rounded-full transition-all duration-200',
              index <= stage ? 'bg-primary' : 'bg-muted',
              index === stage && 'scale-125'
            )}
          />
        ))}
      </div>
    </div>
  );
};

// Smart loading state manager
export const useSmartLoading = (loadingStages: string[], duration = 3000) => {
  const [currentStage, setCurrentStage] = useState(0);
  const [progress, setProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (isComplete) return;

    const stageInterval = duration / loadingStages.length;
    const progressInterval = 50; // Update every 50ms for smooth animation

    const timer = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + (100 / (duration / progressInterval));
        
        if (newProgress >= 100) {
          setIsComplete(true);
          clearInterval(timer);
          return 100;
        }

        // Update stage based on progress
        const newStage = Math.floor((newProgress / 100) * loadingStages.length);
        if (newStage !== currentStage && newStage < loadingStages.length) {
          setCurrentStage(newStage);
        }

        return newProgress;
      });
    }, progressInterval);

    return () => clearInterval(timer);
  }, [loadingStages.length, duration, currentStage, isComplete]);

  const reset = () => {
    setCurrentStage(0);
    setProgress(0);
    setIsComplete(false);
  };

  return {
    currentStage,
    progress,
    isComplete,
    reset
  };
};

// Simulation-specific loading component
export const SimulationLoader = ({ 
  isLoading, 
  simulationType = 'physics' 
}: { 
  isLoading: boolean;
  simulationType?: string;
}) => {
  const stages = [
    'Initializing Environment',
    'Loading Science Engine',
    'Preparing Visualization',
    'Simulation Ready'
  ];

  const { currentStage, progress, isComplete } = useSmartLoading(stages, 2500);

  if (!isLoading || isComplete) return null;

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center">
      <div className="bg-card rounded-lg shadow-lg border border-border p-6 sm:p-8 max-w-sm w-full mx-4">
        <ProgressiveLoader 
          stage={currentStage}
          progress={progress}
          stages={stages}
        />
        <div className="mt-4 text-center">
          <p className="text-xs text-muted-foreground">
            Preparing {simulationType} simulation
          </p>
        </div>
      </div>
    </div>
  );
};

// Legacy alias for backwards compatibility
export const AdvancedLoading = SimulationLoader;