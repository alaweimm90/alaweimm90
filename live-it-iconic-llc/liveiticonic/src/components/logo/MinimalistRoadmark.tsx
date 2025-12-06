import React from 'react';

interface LogoProps {
  className?: string;
  size?: number;
}

/**
 * Minimalist Roadmark - Abstract road/motion inspired logo
 * Represents journey, movement, and the open road
 */
export const MinimalistRoadmark: React.FC<LogoProps> = ({
  className = '',
  size = 100,
}) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Live It Iconic Roadmark"
    >
      <defs>
        <linearGradient id="roadGold" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="hsl(var(--lii-gold))" />
          <stop offset="100%" stopColor="hsl(var(--lii-champagne))" />
        </linearGradient>
      </defs>
      
      {/* Curved road lines - perspective */}
      <path
        d="M 30 80 Q 40 50 50 20"
        stroke="url(#roadGold)"
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
      />
      
      <path
        d="M 70 80 Q 60 50 50 20"
        stroke="url(#roadGold)"
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
      />
      
      {/* Center dashed line */}
      <path
        d="M 50 75 L 50 65 M 50 58 L 50 48 M 50 41 L 50 31 M 50 24 L 50 20"
        stroke="hsl(var(--lii-cloud))"
        strokeWidth="2"
        strokeLinecap="round"
        opacity="0.6"
      />
      
      {/* Horizon accent */}
      <line
        x1="20"
        y1="20"
        x2="80"
        y2="20"
        stroke="url(#roadGold)"
        strokeWidth="1.5"
        opacity="0.4"
      />
      
      {/* Speed lines */}
      <line x1="15" y1="70" x2="25" y2="70" stroke="url(#roadGold)" strokeWidth="1.5" opacity="0.3" />
      <line x1="75" y1="70" x2="85" y2="70" stroke="url(#roadGold)" strokeWidth="1.5" opacity="0.3" />
      <line x1="10" y1="55" x2="22" y2="55" stroke="url(#roadGold)" strokeWidth="1" opacity="0.3" />
      <line x1="78" y1="55" x2="90" y2="55" stroke="url(#roadGold)" strokeWidth="1" opacity="0.3" />
    </svg>
  );
};

export default MinimalistRoadmark;
