import React, { memo, useState, useEffect, useRef } from 'react';
import { cn } from '@/lib/utils';

interface RippleEffectProps {
  children: React.ReactNode;
  className?: string;
  duration?: number;
  color?: string;
}

export const RippleEffect = memo(({ 
  children, 
  className, 
  duration = 600,
  color = "hsl(var(--primary) / 0.3)"
}: RippleEffectProps) => {
  const [ripples, setRipples] = useState<Array<{
    id: number;
    x: number;
    y: number;
    size: number;
  }>>([]);
  const nextRippleId = useRef(0);

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = e.clientX - rect.left - size / 2;
    const y = e.clientY - rect.top - size / 2;
    
    const newRipple = {
      id: nextRippleId.current++,
      x,
      y,
      size
    };
    
    setRipples(prev => [...prev, newRipple]);
    
    setTimeout(() => {
      setRipples(prev => prev.filter(ripple => ripple.id !== newRipple.id));
    }, duration);
  };

  return (
    <div 
      className={cn("relative overflow-hidden", className)}
      onClick={handleClick}
    >
      {children}
      {ripples.map(ripple => (
        <span
          key={ripple.id}
          className="absolute rounded-full animate-ping opacity-0"
          style={{
            left: ripple.x,
            top: ripple.y,
            width: ripple.size,
            height: ripple.size,
            backgroundColor: color,
            animation: `ripple ${duration}ms cubic-bezier(0.4, 0, 0.2, 1)`,
          }}
        />
      ))}
      <style>{`
        @keyframes ripple {
          0% {
            transform: scale(0);
            opacity: 0.6;
          }
          100% {
            transform: scale(2);
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
});

RippleEffect.displayName = 'RippleEffect';

interface MagneticHoverProps {
  children: React.ReactNode;
  className?: string;
  strength?: number;
  resetOnLeave?: boolean;
}

export const MagneticHover = memo(({ 
  children, 
  className, 
  strength = 0.3,
  resetOnLeave = true 
}: MagneticHoverProps) => {
  const elementRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!elementRef.current) return;
    
    const rect = elementRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    const deltaX = (e.clientX - centerX) * strength;
    const deltaY = (e.clientY - centerY) * strength;
    
    elementRef.current.style.transform = `translate(${deltaX}px, ${deltaY}px)`;
  };

  const handleMouseLeave = () => {
    if (!elementRef.current || !resetOnLeave) return;
    elementRef.current.style.transform = 'translate(0px, 0px)';
  };

  return (
    <div
      ref={elementRef}
      className={cn("transition-transform duration-200 ease-out", className)}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {children}
    </div>
  );
});

MagneticHover.displayName = 'MagneticHover';

interface ParallaxElementProps {
  children: React.ReactNode;
  className?: string;
  speed?: number;
  direction?: 'up' | 'down' | 'left' | 'right';
}

export const ParallaxElement = memo(({ 
  children, 
  className, 
  speed = 0.5,
  direction = 'up'
}: ParallaxElementProps) => {
  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      if (!elementRef.current) return;
      
      const rect = elementRef.current.getBoundingClientRect();
      const scrolled = window.pageYOffset;
      const rate = scrolled * speed;
      
      let transform = '';
      switch (direction) {
        case 'up':
          transform = `translateY(-${rate}px)`;
          break;
        case 'down':
          transform = `translateY(${rate}px)`;
          break;
        case 'left':
          transform = `translateX(-${rate}px)`;
          break;
        case 'right':
          transform = `translateX(${rate}px)`;
          break;
      }
      
      elementRef.current.style.transform = transform;
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [speed, direction]);

  return (
    <div ref={elementRef} className={className}>
      {children}
    </div>
  );
});

ParallaxElement.displayName = 'ParallaxElement';

interface FloatingElementProps {
  children: React.ReactNode;
  className?: string;
  intensity?: 'subtle' | 'medium' | 'strong';
  duration?: number;
}

export const FloatingElement = memo(({ 
  children, 
  className, 
  intensity = 'medium',
  duration = 3000 
}: FloatingElementProps) => {
  const intensityMap = {
    subtle: '2px',
    medium: '4px',
    strong: '8px'
  };

  const floatAnimation = `float-${intensity}`;
  
  return (
    <div 
      className={cn("animate-float", className)}
      style={{
        animation: `${floatAnimation} ${duration}ms ease-in-out infinite`,
        '--float-distance': intensityMap[intensity]
      } as React.CSSProperties}
    >
      {children}
      <style>{`
        @keyframes float-subtle {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-2px); }
        }
        @keyframes float-medium {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-4px); }
        }
        @keyframes float-strong {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-8px); }
        }
      `}</style>
    </div>
  );
});

FloatingElement.displayName = 'FloatingElement';

interface GlowOnHoverProps {
  children: React.ReactNode;
  className?: string;
  glowColor?: string;
  glowIntensity?: 'subtle' | 'medium' | 'strong';
}

export const GlowOnHover = memo(({ 
  children, 
  className, 
  glowColor = 'hsl(var(--primary))',
  glowIntensity = 'medium'
}: GlowOnHoverProps) => {
  const intensityMap = {
    subtle: '0 0 10px',
    medium: '0 0 20px',
    strong: '0 0 30px'
  };

  return (
    <div 
      className={cn(
        "transition-all duration-300 hover:scale-[1.02]",
        className
      )}
      style={{
        '--glow-color': glowColor,
        '--glow-intensity': intensityMap[glowIntensity]
      } as React.CSSProperties}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow = `var(--glow-intensity) var(--glow-color)`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = 'none';
      }}
    >
      {children}
    </div>
  );
});

GlowOnHover.displayName = 'GlowOnHover';

interface ScaleOnHoverProps {
  children: React.ReactNode;
  className?: string;
  scale?: number;
  duration?: number;
}

export const ScaleOnHover = memo(({ 
  children, 
  className, 
  scale = 1.05,
  duration = 200
}: ScaleOnHoverProps) => {
  return (
    <div 
      className={cn("cursor-pointer", className)}
      style={{
        transition: `transform ${duration}ms cubic-bezier(0.4, 0, 0.2, 1)`,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = `scale(${scale})`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'scale(1)';
      }}
    >
      {children}
    </div>
  );
});

ScaleOnHover.displayName = 'ScaleOnHover';

interface StaggeredChildrenProps {
  children: React.ReactNode;
  className?: string;
  staggerDelay?: number;
}

export const StaggeredChildren = memo(({ 
  children, 
  className, 
  staggerDelay = 100 
}: StaggeredChildrenProps) => {
  return (
    <div className={cn("stagger-children", className)}>
      {React.Children.map(children, (child, index) => (
        <div style={{ '--stagger-index': index, '--stagger-delay': `${staggerDelay}ms` } as React.CSSProperties}>
          {child}
        </div>
      ))}
    </div>
  );
});

StaggeredChildren.displayName = 'StaggeredChildren';

interface SlideInViewProps {
  children: React.ReactNode;
  className?: string;
  direction?: 'up' | 'down' | 'left' | 'right';
  delay?: number;
}

export const SlideInView = memo(({ 
  children, 
  className, 
  direction = 'up',
  delay = 0 
}: SlideInViewProps) => {
  const [isVisible, setIsVisible] = useState(false);
  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setTimeout(() => setIsVisible(true), delay);
        }
      },
      { threshold: 0.1 }
    );

    if (elementRef.current) {
      observer.observe(elementRef.current);
    }

    return () => observer.disconnect();
  }, [delay]);

  const directionClasses = {
    up: isVisible ? 'translate-y-0 opacity-100' : 'translate-y-8 opacity-0',
    down: isVisible ? 'translate-y-0 opacity-100' : '-translate-y-8 opacity-0',
    left: isVisible ? 'translate-x-0 opacity-100' : 'translate-x-8 opacity-0',
    right: isVisible ? 'translate-x-0 opacity-100' : '-translate-x-8 opacity-0',
  };

  return (
    <div 
      ref={elementRef}
      className={cn(
        "transition-all duration-700 ease-out",
        directionClasses[direction],
        className
      )}
    >
      {children}
    </div>
  );
});

SlideInView.displayName = 'SlideInView';