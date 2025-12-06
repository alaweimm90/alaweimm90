import React from 'react';

interface LogoProps {
  className?: string;
  color?: string;
  size?: number;
}

/**
 * Tropical Tanager Logo - Refined perched bird silhouette
 * Features: Compact profile, alert posture, sophisticated simplicity
 * Ideal for minimalist luxury branding on premium apparel
 */
export const TropicalTanager: React.FC<LogoProps> = ({
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
      aria-label="Tropical Tanager Logo"
    >
      {/* Tanager perched - compact and elegant profile */}
      <path
        d="M 28 12 Q 26 12 24 14 L 22 16 Q 20 18 20 20 L 20 22 Q 20 24 22 26 L 24 28 Q 26 30 28 30 L 30 30 Q 32 30 34 28 L 36 26 Q 38 24 38 22 L 38 20 Q 38 18 36 16 L 34 14 Q 32 12 30 12 L 28 12 Z M 30 30 L 32 32 Q 34 34 34 36 L 34 40 Q 34 44 32 46 L 30 48 Q 28 50 26 50 L 24 50 Q 22 50 20 48 L 18 46 Q 16 44 16 40 L 16 36 Q 16 34 18 32 L 20 30 L 30 30 Z M 26 50 L 24 52 Q 22 54 22 56 L 22 58 Q 22 60 24 62 L 26 64 Q 28 66 30 64 L 32 62 Q 34 60 34 58 L 34 56 Q 34 54 32 52 L 30 50 L 26 50 Z M 28 12 Q 30 10 32 8 L 34 6 Q 36 4 38 4 Q 40 4 42 6 L 44 8 Q 46 10 46 12 L 46 14 Q 46 16 44 18 L 42 20 Q 40 22 38 22 L 38 20 Q 40 18 42 16 L 44 14 Q 46 12 46 10 Q 46 8 44 6 L 42 4 Q 40 2 38 2 Q 36 2 34 4 L 32 6 Q 30 8 28 10 L 28 12 Z"
        className="fill-current stroke-current"
        strokeWidth="0.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

export default TropicalTanager;
