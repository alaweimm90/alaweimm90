import React from 'react';

interface ElegantBackgroundProps {
  className?: string;
  variant?: 'hero' | 'section' | 'minimal' | 'sophisticated' | 'premium';
  colorScheme?: 'warm' | 'cool' | 'mixed' | 'monochrome';
}

export const ElegantBackground: React.FC<ElegantBackgroundProps> = ({ 
  className = '', 
  variant = 'section',
  colorScheme = 'mixed'
}) => {
  const getColorClasses = (scheme: string, opacity: 'light' | 'medium' | 'strong' = 'medium') => {
    const opacityMap = {
      light: { warm: '3', cool: '2', mixed: '4' },
      medium: { warm: '6', cool: '4', mixed: '8' },
      strong: { warm: '10', cool: '8', mixed: '12' }
    };

    const baseOpacity = opacityMap[opacity][scheme as keyof typeof opacityMap.light] || '6';

    switch (scheme) {
      case 'warm':
        return [
          `from-repz-orange/${baseOpacity} to-amber-elegant/${Math.max(2, parseInt(baseOpacity) - 2)}`,
          `from-copper-warm/${baseOpacity} to-gold-luxe/${Math.max(1, parseInt(baseOpacity) - 3)}`,
          `from-amber-elegant/${baseOpacity} to-repz-orange/${Math.max(1, parseInt(baseOpacity) - 4)}`
        ];
      case 'cool':
        return [
          `from-blue-steel/${baseOpacity} to-teal-elegant/${Math.max(2, parseInt(baseOpacity) - 2)}`,
          `from-purple-luxe/${baseOpacity} to-slate-sophisticated/${Math.max(1, parseInt(baseOpacity) - 3)}`,
          `from-teal-elegant/${baseOpacity} to-blue-steel/${Math.max(1, parseInt(baseOpacity) - 4)}`
        ];
      case 'monochrome':
        return [
          `from-silver-muted/${baseOpacity} to-ash-warm/${Math.max(2, parseInt(baseOpacity) - 2)}`,
          `from-graphite-soft/${baseOpacity} to-charcoal-premium/${Math.max(1, parseInt(baseOpacity) - 3)}`,
          `from-ash-warm/${baseOpacity} to-silver-muted/${Math.max(1, parseInt(baseOpacity) - 4)}`
        ];
      default: // mixed
        return [
          `from-repz-orange/${baseOpacity} to-blue-steel/${Math.max(2, parseInt(baseOpacity) - 2)}`,
          `from-purple-luxe/${baseOpacity} to-amber-elegant/${Math.max(1, parseInt(baseOpacity) - 3)}`,
          `from-teal-elegant/${baseOpacity} to-copper-warm/${Math.max(1, parseInt(baseOpacity) - 4)}`
        ];
    }
  };

  const getElements = () => {
    const colors = getColorClasses(colorScheme);
    
    switch (variant) {
      case 'hero':
        return (
          <>
            {/* Large sophisticated background elements */}
            <div className={`absolute top-1/4 left-1/6 w-40 h-40 bg-gradient-to-br ${colors[0]} rounded-full animate-elegant-float backdrop-blur-sm opacity-60`} 
                 style={{animationDelay: '0s', animationDuration: '12s'}}></div>
            <div className={`absolute bottom-1/3 right-1/5 w-32 h-32 bg-gradient-to-br ${colors[1]} rounded-full animate-sophisticated-pulse backdrop-blur-sm opacity-50`} 
                 style={{animationDelay: '2s', animationDuration: '8s'}}></div>
            <div className={`absolute top-2/3 right-1/3 w-24 h-24 bg-gradient-to-br ${colors[2]} rounded-full animate-gentle-drift backdrop-blur-sm opacity-40`} 
                 style={{animationDelay: '4s', animationDuration: '10s'}}></div>
            
            {/* Geometric patterns */}
            <div className="absolute top-1/5 right-2/3 w-16 h-16 border-2 border-repz-orange/15 rotate-45 animate-gentle-pulse backdrop-blur-sm opacity-30" 
                 style={{animationDelay: '1s', animationDuration: '6s'}}></div>
            <div className="absolute bottom-1/4 left-1/3 w-12 h-12 border border-amber-elegant/10 rounded-full animate-gentle-float backdrop-blur-sm opacity-25" 
                 style={{animationDelay: '3s', animationDuration: '8s'}}></div>
          </>
        );
        
      case 'sophisticated':
        return (
          <>
            {/* Medium sophisticated elements with varied animations */}
            <div className={`absolute top-1/6 right-1/4 w-28 h-28 bg-gradient-to-br ${colors[0]} rounded-full animate-sophisticated-pulse backdrop-blur-sm opacity-40`} 
                 style={{animationDelay: '1s', animationDuration: '7s'}}></div>
            <div className={`absolute bottom-1/4 left-1/6 w-20 h-20 bg-gradient-to-br ${colors[1]} rounded-full animate-gentle-drift backdrop-blur-sm opacity-35`} 
                 style={{animationDelay: '3s', animationDuration: '9s'}}></div>
            <div className={`absolute top-1/2 left-3/4 w-16 h-16 bg-gradient-to-br ${colors[2]} rounded-full animate-elegant-float backdrop-blur-sm opacity-30`} 
                 style={{animationDelay: '5s', animationDuration: '11s'}}></div>
          </>
        );
        
      case 'premium':
        return (
          <>
            {/* Premium elements with luxury feel */}
            <div className={`absolute top-1/3 left-1/5 w-36 h-36 bg-gradient-to-br ${colors[0]} rounded-full animate-premium-glow backdrop-blur-sm opacity-50`} 
                 style={{animationDelay: '0s', animationDuration: '15s'}}></div>
            <div className={`absolute bottom-1/3 right-1/4 w-24 h-24 bg-gradient-to-br ${colors[1]} rounded-full animate-elegant-float backdrop-blur-sm opacity-45`} 
                 style={{animationDelay: '2s', animationDuration: '12s'}}></div>
            
            {/* Luxury geometric elements */}
            <div className="absolute top-1/4 right-1/6 w-20 h-20 border-2 border-gold-luxe/20 rotate-12 animate-gentle-pulse backdrop-blur-sm opacity-35" 
                 style={{animationDelay: '1s', animationDuration: '8s'}}></div>
          </>
        );
        
      case 'section':
        return (
          <>
            {/* Medium subtle elements */}
            <div className={`absolute top-1/5 right-1/4 w-24 h-24 bg-gradient-to-br ${colors[0]} rounded-full animate-gentle-float backdrop-blur-sm opacity-35`} 
                 style={{animationDelay: '1s', animationDuration: '8s'}}></div>
            <div className={`absolute bottom-1/4 left-1/6 w-20 h-20 bg-gradient-to-br ${colors[1]} rounded-full animate-gentle-pulse backdrop-blur-sm opacity-30`} 
                 style={{animationDelay: '3s', animationDuration: '10s'}}></div>
          </>
        );
        
      case 'minimal':
        return (
          <>
            {/* Small minimal elements */}
            <div className={`absolute top-1/3 right-1/6 w-16 h-16 bg-gradient-to-br ${colors[0]} rounded-full animate-gentle-drift backdrop-blur-sm opacity-25`} 
                 style={{animationDelay: '2s', animationDuration: '12s'}}></div>
          </>
        );
        
      default:
        return null;
    }
  };

  return (
    <div className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`} style={{zIndex: 0}}>
      {getElements()}
    </div>
  );
};