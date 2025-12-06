// Caribbean Bird Logo Variations - Live It Iconic
// Tropical Hummingbird with Transparent Mountain

import React from 'react';

export const TropicalHummingbird: React.FC<{
  size?: number;
  className?: string;
  label?: string;
}> = ({ size = 100, className = '', label = 'Live It Iconic' }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label={`Live It Iconic - ${label}`}
    >
      <style>{`\n        .lii-champ { stop-color: hsl(var(--lii-champagne)); }\n        .lii-gold { stop-color: hsl(var(--lii-gold)); }\n        .lii-navy { fill: hsl(var(--lii-ink)); }\n        .lii-gold-fill { fill: hsl(var(--lii-gold)); }\n      `}</style>

      <defs>
        <linearGradient id="mountainGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" className="lii-champ" stopOpacity="0.28" />
          <stop offset="50%" className="lii-champ" stopOpacity="0.18" />
          <stop offset="100%" className="lii-champ" stopOpacity="0.08" />
        </linearGradient>

        <linearGradient id="hummingbirdGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" className="lii-gold" />
          <stop offset="100%" className="lii-champ" />
        </linearGradient>

        <filter id="subtleGlow">
          <feGaussianBlur stdDeviation="1.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Mountain Silhouette */}
      <path
        d="M10,75 L25,45 L35,55 L45,35 L55,50 L65,30 L75,45 L85,28 L90,75 Z"
        fill="url(#mountainGrad)"
        opacity="0.5"
      />

      {/* Hummingbird Body */}
      <ellipse
        cx="45"
        cy="35"
        rx="8"
        ry="15"
        fill="url(#hummingbirdGrad)"
        filter="url(#subtleGlow)"
      />

      {/* Wings */}
      <path
        d="M37,30 Q25,20 30,15 Q40,18 37,30"
        className="lii-champ"
        opacity="0.75"
        transform="rotate(-12 37 25)"
      />
      <path
        d="M53,30 Q65,20 70,15 Q60,18 53,30"
        className="lii-champ"
        opacity="0.75"
        transform="rotate(12 53 25)"
      />

      {/* Beak */}
      <path d="M45,20 L45,10 L47,10 Z" className="lii-gold-fill" />

      {/* Motion lines */}
      <g opacity="0.35">
        <line x1="20" y1="35" x2="35" y2="35" stroke="hsl(var(--lii-gold))" strokeWidth="1" />
        <line x1="22" y1="40" x2="37" y2="40" stroke="hsl(var(--lii-gold))" strokeWidth="1" />
        <line x1="18" y1="30" x2="33" y2="30" stroke="hsl(var(--lii-gold))" strokeWidth="1" />
      </g>

      {/* Wordmark */}
      <text
        x="50"
        y="85"
        fontSize="8"
        fill="hsl(var(--lii-ink))"
        textAnchor="middle"
        fontFamily="Montserrat, sans-serif"
        fontWeight="600"
        letterSpacing="0.4px"
      >
        {label}
      </text>
    </svg>
  );
};

export default TropicalHummingbird;
