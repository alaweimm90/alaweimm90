import React, { useEffect, useState } from 'react';
import { useScrollAnimation } from '@/hooks/useScrollAnimation';
import { cn } from '@/lib/utils';

interface ProgressBarProps {
  percentage: number;
  label?: string;
  className?: string;
  color?: 'orange' | 'blue' | 'green' | 'purple';
  showPercentage?: boolean;
  animated?: boolean;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  percentage,
  label,
  className = '',
  color = 'orange',
  showPercentage = true,
  animated = true
}) => {
  const [progress, setProgress] = useState(0);
  const { ref, isVisible } = useScrollAnimation({ threshold: 0.5 });

  useEffect(() => {
    if (isVisible && animated) {
      const timer = setTimeout(() => {
        setProgress(percentage);
      }, 200);
      return () => clearTimeout(timer);
    } else if (!animated) {
      setProgress(percentage);
    }
  }, [isVisible, percentage, animated]);

  const getColorClasses = () => {
    switch (color) {
      case 'orange':
        return 'bg-repz-orange';
      case 'blue':
        return 'bg-gradient-to-r from-blue-500 to-blue-400';
      case 'green':
        return 'bg-gradient-to-r from-green-500 to-green-400';
      case 'purple':
        return 'bg-gradient-to-r from-purple-500 to-purple-400';
      default:
        return 'bg-repz-orange';
    }
  };

  return (
    <div ref={ref as React.RefObject<HTMLDivElement>} className={cn('w-full', className)}>
      {label && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-300">{label}</span>
          {showPercentage && (
            <span className="text-sm font-bold text-repz-orange">{Math.round(progress)}%</span>
          )}
        </div>
      )}
      
      <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
        <div
          className={cn(
            'h-full transition-all duration-1000 ease-out rounded-full',
            getColorClasses()
          )}
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
};