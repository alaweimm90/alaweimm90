import { memo, useEffect, useRef, ReactNode, useState } from 'react';

interface MagneticElementProps {
  children: ReactNode;
  strength?: number;
  className?: string;
  disabled?: boolean;
  attraction?: 'mouse' | 'center' | 'edges';
  glowEffect?: boolean;
  rippleEffect?: boolean;
}

const MagneticElementComponent = ({
  children,
  strength = 0.3,
  className = '',
  disabled = false,
  attraction = 'mouse',
  glowEffect = false,
  rippleEffect = false
}: MagneticElementProps) => {
  const elementRef = useRef<HTMLDivElement>(null);
  const [isHovering, setIsHovering] = useState(false);
  const [ripples, setRipples] = useState<Array<{ id: string; x: number; y: number }>>([]);

  useEffect(() => {
    if (disabled || !elementRef.current) return;

    const element = elementRef.current;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = element.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;

      let targetX = 0;
      let targetY = 0;

      switch (attraction) {
        case 'mouse': {
          targetX = (e.clientX - centerX) * strength;
          targetY = (e.clientY - centerY) * strength;
          break;
        }
        case 'center': {
          const distanceFromCenter = Math.sqrt(
            Math.pow(e.clientX - centerX, 2) + Math.pow(e.clientY - centerY, 2)
          );
          if (distanceFromCenter < rect.width) {
            targetX = -(e.clientX - centerX) * strength * 0.5;
            targetY = -(e.clientY - centerY) * strength * 0.5;
          }
          break;
        }
        case 'edges': {
          if (e.clientX < centerX) {
            targetX = -strength * 20;
          } else {
            targetX = strength * 20;
          }
          if (e.clientY < centerY) {
            targetY = -strength * 20;
          } else {
            targetY = strength * 20;
          }
          break;
        }
      }

      element.style.transform = `translate3d(${targetX}px, ${targetY}px, 0) scale(${1 + strength * 0.1})`;
      
      if (glowEffect) {
        element.style.filter = `brightness(${1 + strength}) drop-shadow(0 0 ${strength * 20}px currentColor)`;
      }
    };

    const handleMouseEnter = () => {
      setIsHovering(true);
      element.style.transition = 'none';
    };

    const handleMouseLeave = () => {
      setIsHovering(false);
      element.style.transition = 'all 0.5s cubic-bezier(0.23, 1, 0.320, 1)';
      element.style.transform = 'translate3d(0, 0, 0) scale(1)';
      if (glowEffect) {
        element.style.filter = 'none';
      }
    };

    const handleClick = (e: MouseEvent) => {
      if (!rippleEffect) return;

      const rect = element.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 100;
      const y = ((e.clientY - rect.top) / rect.height) * 100;

      const rippleId = `ripple-${Date.now()}`;
      setRipples(prev => [...prev, { id: rippleId, x, y }]);

      setTimeout(() => {
        setRipples(prev => prev.filter(r => r.id !== rippleId));
      }, 600);
    };

    element.addEventListener('mouseenter', handleMouseEnter);
    element.addEventListener('mousemove', handleMouseMove);
    element.addEventListener('mouseleave', handleMouseLeave);
    element.addEventListener('click', handleClick);

    return () => {
      element.removeEventListener('mouseenter', handleMouseEnter);
      element.removeEventListener('mousemove', handleMouseMove);
      element.removeEventListener('mouseleave', handleMouseLeave);
      element.removeEventListener('click', handleClick);
    };
  }, [strength, disabled, attraction, glowEffect, rippleEffect]);

  return (
    <div
      ref={elementRef}
      className={`relative cursor-pointer ${className}`}
      style={{
        willChange: 'transform',
        backfaceVisibility: 'hidden',
        perspective: '1000px'
      }}
    >
      {children}
      
      {/* Hover Glow Effect */}
      {glowEffect && isHovering && (
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-secondary/10 rounded-inherit animate-pulse pointer-events-none" />
      )}

      {/* Ripple Effects */}
      {rippleEffect && ripples.map(ripple => (
        <div
          key={ripple.id}
          className="absolute pointer-events-none"
          style={{
            left: `${ripple.x}%`,
            top: `${ripple.y}%`,
            transform: 'translate(-50%, -50%)'
          }}
        >
          <div className="w-0 h-0 bg-primary/30 rounded-full animate-ping" 
               style={{ animation: 'ripple-expand 0.6s ease-out' }} />
        </div>
      ))}

      <style>{`
        @keyframes ripple-expand {
          0% {
            width: 0;
            height: 0;
            opacity: 1;
          }
          100% {
            width: 100px;
            height: 100px;
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
};

export const MagneticElement = memo(MagneticElementComponent);

// Preset magnetic components for common use cases
export const MagneticCard = memo(({ children, className = '', ...props }: Omit<MagneticElementProps, 'strength' | 'glowEffect'>) => (
  <MagneticElement
    strength={0.2}
    glowEffect={true}
    className={`glass-panel hover:shadow-lg transition-shadow duration-300 ${className}`}
    {...props}
  >
    {children}
  </MagneticElement>
));

export const MagneticButton = memo(({ children, className = '', ...props }: Omit<MagneticElementProps, 'strength' | 'rippleEffect'>) => (
  <MagneticElement
    strength={0.4}
    rippleEffect={true}
    className={`relative overflow-hidden ${className}`}
    {...props}
  >
    {children}
  </MagneticElement>
));

export const MagneticIcon = memo(({ children, className = '', ...props }: Omit<MagneticElementProps, 'strength' | 'attraction'>) => (
  <MagneticElement
    strength={0.6}
    attraction="center"
    className={`inline-block ${className}`}
    {...props}
  >
    {children}
  </MagneticElement>
));

// Advanced Hover Effects Component
interface HoverEffectsProps {
  children: ReactNode;
  effect: 'lift' | 'tilt' | 'glow' | 'shake' | 'pulse' | 'rotate' | 'elastic';
  intensity?: 'low' | 'medium' | 'high';
  className?: string;
}

export const HoverEffects = memo(({ 
  children, 
  effect, 
  intensity = 'medium',
  className = '' 
}: HoverEffectsProps) => {
  const getIntensityValue = () => {
    switch (intensity) {
      case 'low': return 0.5;
      case 'medium': return 1;
      case 'high': return 1.5;
      default: return 1;
    }
  };

  const getEffectClasses = () => {
    const scale = getIntensityValue();
    
    switch (effect) {
      case 'lift':
        return `hover:scale-105 hover:-translate-y-2 transition-all duration-300 hover:shadow-xl`;
      case 'tilt':
        return `hover:rotate-1 hover:scale-105 transition-all duration-300`;
      case 'glow':
        return `hover:shadow-glow transition-all duration-300 hover:brightness-110`;
      case 'shake':
        return `hover:animate-pulse transition-all duration-200`;
      case 'pulse':
        return `hover:animate-pulse hover:scale-105 transition-all duration-300`;
      case 'rotate':
        return `hover:rotate-12 hover:scale-110 transition-all duration-300`;
      case 'elastic':
        return `hover:scale-110 transition-all duration-300 hover:animate-bounce`;
      default:
        return '';
    }
  };

  return (
    <div className={`${getEffectClasses()} ${className}`}>
      {children}
    </div>
  );
});

// Micro-interaction component for small UI elements
interface MicroInteractionProps {
  children: ReactNode;
  type: 'button' | 'icon' | 'text' | 'image';
  className?: string;
}

export const MicroInteraction = memo(({ children, type, className = '' }: MicroInteractionProps) => {
  const getTypeClasses = () => {
    switch (type) {
      case 'button':
        return 'hover:scale-105 active:scale-95 transition-transform duration-150 hover:shadow-md';
      case 'icon':
        return 'hover:scale-125 hover:rotate-12 transition-all duration-200 hover:text-primary';
      case 'text':
        return 'hover:text-primary transition-colors duration-200 hover:tracking-wide';
      case 'image':
        return 'hover:scale-105 hover:brightness-110 transition-all duration-300 hover:shadow-lg';
      default:
        return '';
    }
  };

  return (
    <div className={`cursor-pointer ${getTypeClasses()} ${className}`}>
      {children}
    </div>
  );
});

// Interactive Background Elements
export const InteractiveBackground = memo(() => {
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; active: boolean }>>([]);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      const newParticle = {
        id: Date.now(),
        x: e.clientX,
        y: e.clientY,
        active: true
      };

      setParticles(prev => [...prev, newParticle]);

      setTimeout(() => {
        setParticles(prev => prev.filter(p => p.id !== newParticle.id));
      }, 1000);
    };

    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none z-10">
      {particles.map(particle => (
        <div
          key={particle.id}
          className="absolute w-4 h-4 bg-primary/30 rounded-full animate-ping"
          style={{
            left: particle.x - 8,
            top: particle.y - 8,
            animation: 'click-ripple 1s ease-out'
          }}
        />
      ))}
      
      <style>{`
        @keyframes click-ripple {
          0% {
            transform: scale(0);
            opacity: 1;
          }
          100% {
            transform: scale(4);
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
});