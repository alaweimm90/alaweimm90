import React from 'react';

interface LogoProps {
  className?: string;
  size?: number;
}

/**
 * Iconic Diamond Mark - Premium icon-only logo
 * For favicon, social media, app icons
 * Represents precision, luxury, and brilliance
 */
export const IconicDiamondMark: React.FC<LogoProps> = ({
  className = '',
  size = 64,
}) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Live It Iconic Mark"
    >
      <defs>
        <linearGradient id="diamondGold" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="hsl(var(--lii-champagne))" stopOpacity="1" />
          <stop offset="50%" stopColor="hsl(var(--lii-gold))" stopOpacity="1" />
          <stop offset="100%" stopColor="hsl(var(--lii-champagne))" stopOpacity="0.8" />
        </linearGradient>
        
        <filter id="luxuryGlow">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      
      {/* Outer diamond frame */}
      <path
        d="M 32 4 L 56 32 L 32 60 L 8 32 Z"
        stroke="url(#diamondGold)"
        strokeWidth="2"
        fill="none"
        filter="url(#luxuryGlow)"
      />
      
      {/* Inner facets */}
      <path
        d="M 32 4 L 32 32 M 56 32 L 32 32 M 32 60 L 32 32 M 8 32 L 32 32"
        stroke="url(#diamondGold)"
        strokeWidth="1"
        opacity="0.6"
      />
      
      {/* Corner accents */}
      <circle cx="32" cy="4" r="2" fill="url(#diamondGold)" opacity="0.8" />
      <circle cx="56" cy="32" r="2" fill="url(#diamondGold)" opacity="0.8" />
      <circle cx="32" cy="60" r="2" fill="url(#diamondGold)" opacity="0.8" />
      <circle cx="8" cy="32" r="2" fill="url(#diamondGold)" opacity="0.8" />
      
      {/* Center jewel */}
      <circle cx="32" cy="32" r="4" fill="url(#diamondGold)" />
      
      {/* Initials - subtle */}
      <text
        x="32"
        y="36"
        fontSize="10"
        fontWeight="600"
        textAnchor="middle"
        fill="hsl(var(--lii-bg))"
        fontFamily="var(--font-ui)"
        letterSpacing="0.05em"
      >
        LII
      </text>
    </svg>
  );
};

export default IconicDiamondMark;
