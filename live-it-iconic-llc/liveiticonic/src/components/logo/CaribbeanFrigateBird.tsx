import React from 'react';

interface LogoProps {
  className?: string;
  color?: string;
  size?: number;
}

/**
 * Caribbean Frigate Bird Logo - Dynamic silhouette
 * Features: Iconic W-shaped wings, soaring profile, athletic aesthetic
 * Represents freedom and high-performance luxury
 */
export const CaribbeanFrigateBird: React.FC<LogoProps> = ({
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
      aria-label="Caribbean Frigate Bird Logo"
    >
      {/* Frigate bird in flight - distinctive W-wing profile */}
      <path
        d="M 32 16 Q 30 14 28 14 L 24 14 Q 20 14 16 16 L 12 18 Q 8 20 6 24 Q 4 28 4 32 Q 4 34 6 36 L 8 38 Q 10 40 14 42 L 18 44 Q 22 46 26 46 L 30 46 Q 32 46 32 44 L 32 40 Q 32 38 30 36 L 28 34 Q 26 32 26 30 L 26 28 Q 26 26 28 24 L 30 22 Q 32 20 32 18 L 32 16 Z M 32 16 Q 34 14 36 14 L 40 14 Q 44 14 48 16 L 52 18 Q 56 20 58 24 Q 60 28 60 32 Q 60 34 58 36 L 56 38 Q 54 40 50 42 L 46 44 Q 42 46 38 46 L 34 46 Q 32 46 32 44 L 32 40 Q 32 38 34 36 L 36 34 Q 38 32 38 30 L 38 28 Q 38 26 36 24 L 34 22 Q 32 20 32 18 L 32 16 Z M 32 18 L 32 22 Q 32 24 30 26 L 28 28 Q 26 30 26 32 L 26 34 Q 26 36 28 38 L 30 40 Q 32 42 32 44 L 32 48 Q 32 50 34 52 L 36 54 Q 38 56 40 56 Q 42 56 44 54 L 46 52 Q 48 50 48 48 L 48 44 Q 48 42 46 40 L 44 38 Q 42 36 42 34 L 42 32 Q 42 30 44 28 L 46 26 Q 48 24 48 22 L 48 18 Q 46 20 44 22 L 40 26 Q 36 30 32 32 Q 28 30 24 26 L 20 22 Q 18 20 16 18 L 16 22 Q 16 24 18 26 L 20 28 Q 22 30 22 32 L 22 34 Q 22 36 20 38 L 18 40 Q 16 42 16 44 L 16 48 Q 16 50 18 52 L 20 54 Q 22 56 24 56 Q 26 56 28 54 L 30 52 Q 32 50 32 48 L 32 44 Q 32 42 34 40 L 36 38 Q 38 36 38 34 L 38 32 Q 38 30 36 28 L 34 26 Q 32 24 32 22 L 32 18 Z"
        className="fill-current stroke-current"
        strokeWidth="0.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

export default CaribbeanFrigateBird;
