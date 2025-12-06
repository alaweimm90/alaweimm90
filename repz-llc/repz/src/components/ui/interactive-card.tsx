import React, { useState } from 'react';
import { Card, CardContent } from '@/ui/molecules/Card';
import { cn } from '@/lib/utils';

interface InteractiveCardProps {
  children: React.ReactNode;
  className?: string;
  hoverEffect?: 'lift' | 'glow' | 'scale' | 'tilt';
  clickEffect?: boolean;
}

export const InteractiveCard: React.FC<InteractiveCardProps> = ({
  children,
  className = '',
  hoverEffect = 'lift',
  clickEffect = true
}) => {
  const [isClicked, setIsClicked] = useState(false);

  const handleClick = () => {
    if (clickEffect) {
      setIsClicked(true);
      setTimeout(() => setIsClicked(false), 200);
    }
  };

  const getHoverEffectClasses = () => {
    switch (hoverEffect) {
      case 'lift':
        return 'hover:-translate-y-2 hover:shadow-xl';
      case 'glow':
        return 'hover:shadow-lg hover:shadow-repz-orange/20 hover:border-repz-orange/50';
      case 'scale':
        return 'hover:scale-105';
      case 'tilt':
        return 'hover:rotate-1 hover:-translate-y-1';
      default:
        return 'hover:-translate-y-2 hover:shadow-xl';
    }
  };

  return (
    <Card
      className={cn(
        'transition-all duration-300 ease-out cursor-pointer',
        getHoverEffectClasses(),
        isClicked && 'scale-95',
        className
      )}
      onClick={handleClick}
    >
      <CardContent className="relative overflow-hidden">
        {children}
        
        {/* Ripple effect on click */}
        {clickEffect && isClicked && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="w-full h-full bg-repz-orange/20 rounded-lg animate-ping" />
          </div>
        )}
      </CardContent>
    </Card>
  );
};