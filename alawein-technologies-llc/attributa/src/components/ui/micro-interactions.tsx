import React from 'react';
import { cn } from '@/lib/utils';

// Hover lift effect for cards
interface HoverLiftProps {
  children: React.ReactNode;
  className?: string;
  intensity?: 'subtle' | 'medium' | 'strong';
}

export function HoverLift({ children, className, intensity = 'medium' }: HoverLiftProps) {
  const intensityClasses = {
    subtle: 'hover:-translate-y-1 hover:shadow-lg',
    medium: 'hover:-translate-y-2 hover:shadow-xl',
    strong: 'hover:-translate-y-3 hover:shadow-2xl',
  };

  return (
    <div className={cn(
      'transition-all duration-300 ease-out',
      intensityClasses[intensity],
      className
    )}>
      {children}
    </div>
  );
}

// Magnetic button effect
interface MagneticButtonProps {
  children: React.ReactNode;
  className?: string;
  magnetStrength?: number;
}

export function MagneticButton({ children, className, magnetStrength = 0.3 }: MagneticButtonProps) {
  const ref = React.useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!ref.current) return;

    const { left, top, width, height } = ref.current.getBoundingClientRect();
    const centerX = left + width / 2;
    const centerY = top + height / 2;
    
    const deltaX = (e.clientX - centerX) * magnetStrength;
    const deltaY = (e.clientY - centerY) * magnetStrength;

    ref.current.style.transform = `translate(${deltaX}px, ${deltaY}px)`;
  };

  const handleMouseLeave = () => {
    if (ref.current) {
      ref.current.style.transform = 'translate(0px, 0px)';
    }
  };

  return (
    <div
      ref={ref}
      className={cn('transition-transform duration-300 ease-out', className)}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {children}
    </div>
  );
}

// Parallax scroll effect
interface ParallaxProps {
  children: React.ReactNode;
  speed?: number;
  className?: string;
}

export function Parallax({ children, speed = 0.5, className }: ParallaxProps) {
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const handleScroll = () => {
      const scrolled = window.pageYOffset;
      const rate = scrolled * -speed;
      element.style.transform = `translateY(${rate}px)`;
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [speed]);

  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
}

// Stagger animation for lists
interface StaggeredListProps {
  children: React.ReactNode[];
  className?: string;
  staggerDelay?: number;
}

export function StaggeredList({ children, className, staggerDelay = 0.1 }: StaggeredListProps) {
  const [isVisible, setIsVisible] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <div ref={ref} className={className}>
      {children.map((child, index) => (
        <div
          key={index}
          className={cn(
            'transition-all duration-700 ease-out',
            isVisible 
              ? 'opacity-100 translate-y-0' 
              : 'opacity-0 translate-y-8'
          )}
          style={{
            transitionDelay: isVisible ? `${index * staggerDelay}s` : '0s',
          }}
        >
          {child}
        </div>
      ))}
    </div>
  );
}

// Pulse on hover
interface PulseHoverProps {
  children: React.ReactNode;
  className?: string;
  color?: string;
}

export function PulseHover({ children, className, color = 'primary' }: PulseHoverProps) {
  return (
    <div className={cn(
      'relative overflow-hidden transition-all duration-300',
      'hover:scale-[1.02] hover:shadow-lg',
      `hover:shadow-${color}/20`,
      className
    )}>
      {children}
      <div className={cn(
        'absolute inset-0 opacity-0 transition-opacity duration-300',
        'hover:opacity-100 pointer-events-none',
        `bg-${color}/5`
      )} />
    </div>
  );
}

// Smooth reveal on scroll
interface ScrollRevealProps {
  children: React.ReactNode;
  className?: string;
  direction?: 'up' | 'down' | 'left' | 'right';
  distance?: number;
  delay?: number;
}

export function ScrollReveal({ 
  children, 
  className, 
  direction = 'up', 
  distance = 30,
  delay = 0 
}: ScrollRevealProps) {
  const [isVisible, setIsVisible] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);

  const directionTransforms = {
    up: `translateY(${distance}px)`,
    down: `translateY(-${distance}px)`,
    left: `translateX(${distance}px)`,
    right: `translateX(-${distance}px)`,
  };

  React.useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setTimeout(() => setIsVisible(true), delay * 1000);
        }
      },
      { threshold: 0.1 }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => observer.disconnect();
  }, [delay]);

  return (
    <div
      ref={ref}
      className={cn(
        'transition-all duration-700 ease-out',
        isVisible 
          ? 'opacity-100 translate-x-0 translate-y-0' 
          : 'opacity-0',
        className
      )}
      style={{
        transform: isVisible ? 'none' : directionTransforms[direction],
      }}
    >
      {children}
    </div>
  );
}