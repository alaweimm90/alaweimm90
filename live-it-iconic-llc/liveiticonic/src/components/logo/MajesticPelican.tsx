import React from 'react';

interface LogoProps {
  className?: string;
  color?: string;
  size?: number;
}

/**
 * Majestic Pelican Logo - Bold Caribbean bird silhouette
 * Features: Distinctive bill profile, strong presence, premium feel
 * Perfect for luxury automotive lifestyle branding
 */
export const MajesticPelican: React.FC<LogoProps> = ({
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
      aria-label="Majestic Pelican Logo"
    >
      {/* Pelican with distinctive bill - simplified for embroidery */}
      <path
        d="M 12 20 Q 10 22 10 24 L 10 28 Q 10 30 12 32 L 16 34 Q 20 36 24 36 L 28 36 Q 32 36 34 34 L 36 32 Q 38 30 40 28 L 42 26 Q 44 24 44 22 L 44 20 Q 44 18 42 16 L 40 14 Q 38 12 36 12 L 32 12 Q 28 12 26 14 L 24 16 Q 22 18 20 18 L 16 18 Q 14 18 12 20 Z M 28 36 L 30 40 Q 32 44 32 48 L 32 52 Q 32 54 30 56 L 28 58 Q 26 60 24 60 Q 22 60 20 58 L 18 56 Q 16 54 16 52 L 16 48 Q 16 44 18 40 L 20 36 L 28 36 Z M 32 12 Q 34 10 36 8 L 38 6 Q 40 4 42 4 Q 44 4 46 6 L 48 8 Q 50 10 50 12 L 50 16 Q 50 18 48 20 L 46 22 Q 44 24 42 26 L 40 28 Q 42 26 44 24 L 46 22 Q 48 20 48 18 L 48 14 Q 48 12 46 10 L 44 8 Q 42 6 40 6 Q 38 6 36 8 L 34 10 Q 32 12 32 12 Z"
        className="fill-current stroke-current"
        strokeWidth="0.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

export default MajesticPelican;
