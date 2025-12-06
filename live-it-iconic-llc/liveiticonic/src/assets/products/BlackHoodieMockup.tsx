/**
 * Black Hoodie Product Mockup
 * Features: Majestic Pelican logo in championship gold
 * Luxury fleece, embossed details for premium automotive lifestyle
 */

import React from 'react';

interface ProductMockupProps {
  className?: string;
  width?: number;
  height?: number;
}

export const BlackHoodieMockup: React.FC<ProductMockupProps> = ({
  className = '',
  width = 800,
  height = 1000,
}) => {
  return (
    <svg
      width={width}
      height={height}
      viewBox="0 0 800 1000"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* White background */}
      <rect width="800" height="1000" fill="#FFFFFF" />

      {/* Hoodie silhouette - carbon black */}
      <g id="hoodie">
        {/* Main body with kangaroo pocket */}
        <path
          d="M 250 280 L 200 320 L 200 750 Q 200 800 250 850 L 550 850 Q 600 800 600 750 L 600 320 L 550 280 L 520 220 Q 480 180 400 180 Q 320 180 280 220 L 250 280 Z"
          fill="#1a1a1a"
          stroke="#0a0a0a"
          strokeWidth="2"
        />

        {/* Kangaroo pocket */}
        <rect
          x="300"
          y="520"
          width="200"
          height="120"
          rx="10"
          fill="#0f0f0f"
          stroke="#050505"
          strokeWidth="2"
        />

        {/* Hood */}
        <path
          d="M 280 220 Q 300 150 400 150 Q 500 150 520 220 L 500 240 Q 480 200 400 200 Q 320 200 300 240 L 280 220 Z"
          fill="#1a1a1a"
          stroke="#0a0a0a"
          strokeWidth="2"
        />

        {/* Drawstrings */}
        <circle cx="350" cy="250" r="4" fill="#d4af37" />
        <circle cx="450" cy="250" r="4" fill="#d4af37" />
        <path
          d="M 350 250 Q 380 260 400 260 Q 420 260 450 250"
          stroke="#d4af37"
          strokeWidth="2"
          fill="none"
        />

        {/* Sleeves */}
        <path
          d="M 250 280 L 150 340 Q 100 360 100 400 L 120 450 Q 140 480 180 465 L 200 440 L 200 320 L 250 280 Z"
          fill="#1a1a1a"
          stroke="#0a0a0a"
          strokeWidth="2"
        />
        <path
          d="M 550 280 L 650 340 Q 700 360 700 400 L 680 450 Q 660 480 620 465 L 600 440 L 600 320 L 550 280 Z"
          fill="#1a1a1a"
          stroke="#0a0a0a"
          strokeWidth="2"
        />

        {/* Ribbed cuffs (subtle detail) */}
        <rect x="190" y="440" width="20" height="10" rx="2" fill="#0f0f0f" opacity="0.5" />
        <rect x="590" y="440" width="20" height="10" rx="2" fill="#0f0f0f" opacity="0.5" />
      </g>

      {/* Majestic Pelican Logo - Championship Gold - chest placement */}
      <g id="logo" transform="translate(300, 340)">
        <path
          d="M 30 50 Q 25 55 25 60 L 25 70 Q 25 75 30 80 L 40 85 Q 50 90 60 90 L 70 90 Q 80 90 85 85 L 90 80 Q 95 75 100 70 L 105 65 Q 110 60 110 55 L 110 50 Q 110 45 105 40 L 100 35 Q 95 30 90 30 L 80 30 Q 70 30 65 35 L 60 40 Q 55 45 50 45 L 40 45 Q 35 45 30 50 Z M 70 90 L 75 100 Q 80 110 80 120 L 80 130 Q 80 135 75 140 L 70 145 Q 65 150 60 150 Q 55 150 50 145 L 45 140 Q 40 135 40 130 L 40 120 Q 40 110 45 100 L 50 90 L 70 90 Z M 80 30 Q 85 25 90 20 L 95 15 Q 100 10 105 10 Q 110 10 115 15 L 120 20 Q 125 25 125 30 L 125 40 Q 125 45 120 50 L 115 55 Q 110 60 105 65 L 100 70 Q 105 65 110 60 L 115 55 Q 120 50 120 45 L 120 35 Q 120 30 115 25 L 110 20 Q 105 15 100 15 Q 95 15 90 20 L 85 25 Q 80 30 80 30 Z"
          fill="#d4af37"
          stroke="#d4af37"
          strokeWidth="1"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>

      {/* Subtle shadow for depth */}
      <ellipse cx="400" cy="900" rx="220" ry="35" fill="#000000" opacity="0.1" />

      {/* Product details text */}
      <text
        x="400"
        y="960"
        fontFamily="Helvetica, sans-serif"
        fontSize="24"
        fontWeight="bold"
        fill="#1a1a1a"
        textAnchor="middle"
      >
        LIVE IT ICONIC
      </text>
      <text
        x="400"
        y="985"
        fontFamily="Helvetica, sans-serif"
        fontSize="16"
        fill="#666666"
        textAnchor="middle"
      >
        Premium Carbon Black Hoodie
      </text>
    </svg>
  );
};

export default BlackHoodieMockup;
