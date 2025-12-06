import React from 'react';

interface SectionBoundaryProps {
  variant?: 'top' | 'bottom' | 'both' | 'hero' | 'premium';
  className?: string;
  glowEffect?: boolean;
  animated?: boolean;
}

export const SectionBoundary: React.FC<SectionBoundaryProps> = ({ 
  variant = 'top', 
  className = '',
  glowEffect = false,
  animated = true
}) => {
  const getBoundaryContent = () => {
    const baseClasses = "w-full relative overflow-hidden";
    const glowClasses = glowEffect ? "shadow-glow-orange" : "";
    const animationClasses = animated ? "animate-gradient-shift" : "";
    
    switch (variant) {
      case 'hero':
        return (
          <div className={`${baseClasses} h-2 ${className}`}>
            <div className={`absolute inset-0 bg-gradient-to-r from-transparent via-repz-orange/60 via-amber-elegant/50 to-transparent ${animationClasses} ${glowClasses}`}></div>
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-gold-luxe/30 to-transparent opacity-70"></div>
          </div>
        );
        
      case 'premium':
        return (
          <div className={`${baseClasses} h-1.5 ${className}`}>
            <div className={`absolute inset-0 bg-gradient-to-r from-transparent via-purple-luxe/50 via-repz-orange/60 via-blue-steel/40 to-transparent ${animationClasses} ${glowClasses}`}></div>
            <div className="absolute top-0 left-1/4 right-1/4 h-0.5 bg-gradient-to-r from-gold-luxe/80 to-copper-warm/60"></div>
          </div>
        );
        
      default:
        return (
          <div className={`${baseClasses} h-1 ${className}`}>
            <div className={`absolute inset-0 bg-gradient-to-r from-transparent via-repz-orange/50 to-transparent ${animationClasses} ${glowClasses}`}></div>
          </div>
        );
    }
  };

  const borderClass = "border-repz-orange/30";
  const borderThickness = variant === 'hero' ? 'border-t-8' : variant === 'premium' ? 'border-t-6' : 'border-t-4';

  switch (variant) {
    case 'top':
      return (
        <div className={`${borderThickness} ${borderClass} relative`}>
          {getBoundaryContent()}
        </div>
      );
    case 'bottom':
      return (
        <div className={`border-b-4 ${borderClass} relative`}>
          {getBoundaryContent()}
        </div>
      );
    case 'both':
      return (
        <div className={`border-t-4 border-b-4 ${borderClass} relative`}>
          {getBoundaryContent()}
        </div>
      );
    case 'hero':
    case 'premium':
      return (
        <div className={`${borderThickness} ${borderClass} relative`}>
          {getBoundaryContent()}
        </div>
      );
    default:
      return getBoundaryContent();
  }
};