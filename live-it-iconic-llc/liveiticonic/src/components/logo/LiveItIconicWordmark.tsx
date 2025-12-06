import React from 'react';

interface LogoProps {
  className?: string;
  size?: number;
  variant?: 'horizontal' | 'stacked';
}

/**
 * Live It Iconic - Primary Wordmark Logo
 * Luxury automotive lifestyle brand
 * Minimalist, confident, precise
 */
export const LiveItIconicWordmark: React.FC<LogoProps> = ({
  className = '',
  size = 200,
  variant = 'horizontal',
}) => {
  if (variant === 'stacked') {
    return (
      <svg
        width={size}
        height={size * 1.2}
        viewBox="0 0 200 240"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className={className}
        aria-label="Live It Iconic"
      >
        <defs>
          <linearGradient id="goldGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="hsl(var(--lii-gold))" />
            <stop offset="100%" stopColor="hsl(var(--lii-champagne))" />
          </linearGradient>
        </defs>
        
        {/* LIVE */}
        <text
          x="100"
          y="60"
          fontSize="48"
          fontWeight="300"
          letterSpacing="0.15em"
          textAnchor="middle"
          fill="hsl(var(--lii-cloud))"
          fontFamily="var(--font-display)"
        >
          LIVE
        </text>
        
        {/* IT */}
        <text
          x="100"
          y="120"
          fontSize="48"
          fontWeight="600"
          letterSpacing="0.15em"
          textAnchor="middle"
          fill="url(#goldGradient)"
          fontFamily="var(--font-display)"
        >
          IT
        </text>
        
        {/* ICONIC */}
        <text
          x="100"
          y="180"
          fontSize="48"
          fontWeight="300"
          letterSpacing="0.15em"
          textAnchor="middle"
          fill="hsl(var(--lii-cloud))"
          fontFamily="var(--font-display)"
        >
          ICONIC
        </text>
        
        {/* Accent line */}
        <line
          x1="50"
          y1="200"
          x2="150"
          y2="200"
          stroke="url(#goldGradient)"
          strokeWidth="2"
          strokeLinecap="round"
        />
      </svg>
    );
  }

  return (
    <svg
      width={size}
      height={size * 0.3}
      viewBox="0 0 600 180"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Live It Iconic"
    >
      <defs>
        <linearGradient id="goldGradientH" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="hsl(var(--lii-gold))" />
          <stop offset="100%" stopColor="hsl(var(--lii-champagne))" />
        </linearGradient>
      </defs>
      
      {/* Diamond icon mark */}
      <path
        d="M 30 60 L 50 90 L 30 120 L 10 90 Z"
        stroke="url(#goldGradientH)"
        strokeWidth="2"
        fill="none"
      />
      <circle cx="30" cy="90" r="3" fill="url(#goldGradientH)" />
      
      {/* LIVE */}
      <text
        x="80"
        y="110"
        fontSize="56"
        fontWeight="300"
        letterSpacing="0.12em"
        fill="hsl(var(--lii-cloud))"
        fontFamily="var(--font-display)"
      >
        LIVE
      </text>
      
      {/* IT */}
      <text
        x="240"
        y="110"
        fontSize="56"
        fontWeight="600"
        letterSpacing="0.15em"
        fill="url(#goldGradientH)"
        fontFamily="var(--font-display)"
      >
        IT
      </text>
      
      {/* ICONIC */}
      <text
        x="350"
        y="110"
        fontSize="56"
        fontWeight="300"
        letterSpacing="0.12em"
        fill="hsl(var(--lii-cloud))"
        fontFamily="var(--font-display)"
      >
        ICONIC
      </text>
      
      {/* Subtle underline accent */}
      <line
        x1="80"
        y1="130"
        x2="580"
        y2="130"
        stroke="url(#goldGradientH)"
        strokeWidth="1.5"
        opacity="0.4"
        strokeLinecap="round"
      />
    </svg>
  );
};

export default LiveItIconicWordmark;
