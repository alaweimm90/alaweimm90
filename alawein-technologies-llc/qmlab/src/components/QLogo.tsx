import React from 'react';

interface QLogo {
  className?: string;
  variant?: 'full' | 'mark' | 'text';
}

export const QLogo: React.FC<QLogo> = ({ 
  className = "w-8 h-8", 
  variant = 'mark' 
}) => {
  if (variant === 'text') {
    return (
      <div className={`font-display font-bold text-primary ${className}`}>
        QMLab
      </div>
    );
  }

  if (variant === 'full') {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <QLogo variant="mark" className="w-8 h-8" />
        <span className="font-display font-bold text-text">QMLab</span>
      </div>
    );
  }

  // Mark variant (default)
  return (
    <svg 
      viewBox="0 0 32 32" 
      className={className}
      role="img"
      aria-label="QMLab Logo"
    >
      {/* Outer quantum gate frame */}
      <rect 
        x="2" 
        y="2" 
        width="28" 
        height="28" 
        rx="6" 
        fill="none" 
        stroke="hsl(var(--primary))" 
        strokeWidth="2"
      />
      
      {/* Q letter with quantum circuit styling */}
      <circle 
        cx="16" 
        cy="14" 
        r="6" 
        fill="none" 
        stroke="hsl(var(--primary))" 
        strokeWidth="2.5"
      />
      
      {/* Q tail - quantum entanglement line */}
      <path 
        d="M20 18 L24 22" 
        stroke="hsl(var(--primary))" 
        strokeWidth="2.5" 
        strokeLinecap="round"
      />
      
      {/* Quantum state indicators */}
      <circle 
        cx="12" 
        cy="10" 
        r="1" 
        fill="hsl(var(--state-pure))"
        className="animate-pulse"
      />
      <circle 
        cx="20" 
        cy="18" 
        r="1" 
        fill="hsl(var(--state-entangled))"
        className="animate-pulse"
        style={{ animationDelay: '0.5s' }}
      />
      
      {/* Subtle glow effect */}
      <circle 
        cx="16" 
        cy="14" 
        r="8" 
        fill="hsl(var(--primary))" 
        opacity="0.1"
        className="animate-quantum-pulse"
      />
    </svg>
  );
};