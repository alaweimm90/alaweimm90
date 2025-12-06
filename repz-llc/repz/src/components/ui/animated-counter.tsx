import React, { useEffect, useState } from 'react';
import { useScrollAnimation } from '@/hooks/useScrollAnimation';

interface AnimatedCounterProps {
  end: number;
  duration?: number;
  suffix?: string;
  prefix?: string;
  className?: string;
}

export const AnimatedCounter: React.FC<AnimatedCounterProps> = ({
  end,
  duration = 2000,
  suffix = '',
  prefix = '',
  className = ''
}) => {
  const [count, setCount] = useState(0);
  const { ref, isVisible } = useScrollAnimation({ threshold: 0.5 });

  useEffect(() => {
    if (isVisible) {
      let startTime: number;
      const startCount = 0;
      
      const updateCount = (timestamp: number) => {
        if (!startTime) startTime = timestamp;
        const progress = Math.min((timestamp - startTime) / duration, 1);
        
        // Easing function for smooth animation
        const easeOutCubic = 1 - Math.pow(1 - progress, 3);
        const currentCount = Math.floor(easeOutCubic * (end - startCount) + startCount);
        
        setCount(currentCount);
        
        if (progress < 1) {
          requestAnimationFrame(updateCount);
        } else {
          setCount(end);
        }
      };
      
      requestAnimationFrame(updateCount);
    }
  }, [isVisible, end, duration]);

  return (
    <div ref={ref as React.RefObject<HTMLDivElement>} className={className}>
      <span>{prefix}{count.toLocaleString()}{suffix}</span>
    </div>
  );
};