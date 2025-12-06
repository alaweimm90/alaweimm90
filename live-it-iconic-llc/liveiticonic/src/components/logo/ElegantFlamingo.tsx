import React from 'react';

interface LogoProps {
  className?: string;
  color?: string;
  size?: number;
}

/**
 * Elegant Flamingo Logo - Minimalist Caribbean bird silhouette
 * Inspired by iconic luxury brand aesthetics (Lacoste, Polo)
 * Features: Graceful S-curve neck, single leg stance, refined profile
 */
export const ElegantFlamingo: React.FC<LogoProps> = ({
  className = '',
  color = 'currentColor',
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
      aria-label="Elegant Flamingo Logo"
    >
      {/* Minimalist flamingo silhouette - single path for clean embroidery */}
      <path
        d="M 32 8 Q 35 10 36 14 Q 36 18 34 20 L 32 22 Q 30 23 28 24 L 26 26 Q 24 28 24 32 L 24 38 Q 24 42 26 44 L 28 46 Q 30 48 32 48 Q 33 48 34 49 L 35 52 Q 35 54 34 56 L 33 58 Q 32 60 30 60 Q 28 60 27 58 L 26 52 Q 26 48 28 46 L 29 44 Q 27 42 26 40 L 24 36 Q 22 32 22 28 L 22 24 Q 22 20 24 18 L 26 16 Q 28 14 30 13 L 32 12 Q 30 10 28 9 L 26 8 Q 24 7 22 8 L 20 10 Q 18 12 18 14 L 18 16 Q 20 16 22 17 L 24 18 Q 22 18 20 17 L 18 16 Q 16 14 16 12 L 16 10 Q 16 8 18 6 L 20 4 Q 22 3 24 3 L 28 4 Q 30 5 32 8 Z"
        className="fill-current stroke-current"
        strokeWidth="0.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

export default ElegantFlamingo;
