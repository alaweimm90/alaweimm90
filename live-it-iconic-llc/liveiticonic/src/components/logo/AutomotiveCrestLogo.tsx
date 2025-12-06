import React from 'react';

interface LogoProps {
  className?: string;
  size?: number;
}

/**
 * Automotive Crest Logo - Premium badge inspired by luxury car emblems
 * Heritage meets modern performance
 */
export const AutomotiveCrestLogo: React.FC<LogoProps> = ({
  className = '',
  size = 120,
}) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 120 120"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Live It Iconic Crest"
    >
      <defs>
        <linearGradient id="crestGold" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="hsl(var(--lii-champagne))" />
          <stop offset="50%" stopColor="hsl(var(--lii-gold))" />
          <stop offset="100%" stopColor="hsl(var(--lii-gold-press))" />
        </linearGradient>
        
        <linearGradient id="crestSilver" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="hsl(var(--lii-cloud))" />
          <stop offset="100%" stopColor="hsl(var(--lii-ash))" />
        </linearGradient>
      </defs>
      
      {/* Shield outline */}
      <path
        d="M 60 10 L 90 30 L 90 70 Q 90 90 60 105 Q 30 90 30 70 L 30 30 Z"
        stroke="url(#crestGold)"
        strokeWidth="2.5"
        fill="hsl(var(--lii-ink))"
      />
      
      {/* Inner shield detail */}
      <path
        d="M 60 18 L 82 34 L 82 68 Q 82 84 60 97 Q 38 84 38 68 L 38 34 Z"
        stroke="url(#crestSilver)"
        strokeWidth="1"
        fill="none"
        opacity="0.3"
      />
      
      {/* Center divider */}
      <line
        x1="60"
        y1="25"
        x2="60"
        y2="95"
        stroke="url(#crestGold)"
        strokeWidth="1.5"
        opacity="0.4"
      />
      
      {/* L */}
      <path
        d="M 42 45 L 42 70 L 52 70"
        stroke="url(#crestGold)"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      
      {/* I */}
      <line
        x1="60"
        y1="45"
        x2="60"
        y2="70"
        stroke="url(#crestGold)"
        strokeWidth="3"
        strokeLinecap="round"
      />
      
      {/* I */}
      <line
        x1="68"
        y1="45"
        x2="68"
        y2="70"
        stroke="url(#crestGold)"
        strokeWidth="3"
        strokeLinecap="round"
      />
      <circle cx="68" cy="40" r="2" fill="url(#crestGold)" />
      
      {/* Top accent */}
      <circle cx="60" cy="15" r="3" fill="url(#crestGold)" />
      
      {/* Corner stars */}
      <circle cx="35" cy="35" r="1.5" fill="url(#crestGold)" opacity="0.6" />
      <circle cx="85" cy="35" r="1.5" fill="url(#crestGold)" opacity="0.6" />
    </svg>
  );
};

export default AutomotiveCrestLogo;
