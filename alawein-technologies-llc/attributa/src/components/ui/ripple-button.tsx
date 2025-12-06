import React, { useState } from 'react';
import { cn } from '@/lib/utils';

interface RippleButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: 'default' | 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function RippleButton({ 
  children, 
  variant = 'default', 
  size = 'md', 
  className,
  onMouseDown,
  ...props 
}: RippleButtonProps) {
  const [ripples, setRipples] = useState<Array<{ id: number; x: number; y: number }>>([]);

  const handleMouseDown = (e: React.MouseEvent<HTMLButtonElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const newRipple = {
      id: Date.now(),
      x,
      y
    };

    setRipples(prev => [...prev, newRipple]);

    // Remove ripple after animation
    setTimeout(() => {
      setRipples(prev => prev.filter(ripple => ripple.id !== newRipple.id));
    }, 600);

    onMouseDown?.(e);
  };

  const variants = {
    default: "bg-background border border-border text-foreground hover:bg-accent",
    primary: "bg-primary text-primary-foreground hover:bg-primary/90",
    secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
    ghost: "text-foreground hover:bg-accent hover:text-accent-foreground"
  };

  const sizes = {
    sm: "h-8 px-3 text-sm",
    md: "h-10 px-4 text-sm",
    lg: "h-12 px-6 text-base"
  };

  return (
    <button
      className={cn(
        "relative overflow-hidden rounded-md font-medium transition-all duration-200",
        "focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2",
        "active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none",
        variants[variant],
        sizes[size],
        className
      )}
      onMouseDown={handleMouseDown}
      {...props}
    >
      {children}
      
      {/* Ripple effects */}
      {ripples.map((ripple) => (
        <span
          key={ripple.id}
          className="absolute pointer-events-none rounded-full bg-foreground/30 animate-ping"
          style={{
            left: ripple.x - 10,
            top: ripple.y - 10,
            width: 20,
            height: 20,
            animationDuration: '0.6s',
            animationTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)'
          }}
        />
      ))}
    </button>
  );
}