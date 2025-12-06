/**
 * Black T-Shirt Product Mockup
 * Features: Elegant Flamingo logo in championship gold
 * Premium cotton, minimalist design for luxury automotive lifestyle
 */

import React from 'react';

interface ProductMockupProps {
  className?: string;
  width?: number;
  height?: number;
}

export const BlackTShirtMockup: React.FC<ProductMockupProps> = ({
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

      {/* T-shirt silhouette - carbon black */}
      <g id="t-shirt">
        {/* Main body */}
        <path
          d="M 250 250 L 200 300 L 200 700 Q 200 750 250 800 L 550 800 Q 600 750 600 700 L 600 300 L 550 250 L 500 200 Q 450 150 400 150 Q 350 150 300 200 L 250 250 Z"
          fill="#1a1a1a"
          stroke="#0a0a0a"
          strokeWidth="2"
        />

        {/* Sleeves */}
        <path
          d="M 250 250 L 150 300 Q 100 320 100 350 L 120 380 Q 140 400 180 390 L 200 370 L 200 300 L 250 250 Z"
          fill="#1a1a1a"
          stroke="#0a0a0a"
          strokeWidth="2"
        />
        <path
          d="M 550 250 L 650 300 Q 700 320 700 350 L 680 380 Q 660 400 620 390 L 600 370 L 600 300 L 550 250 Z"
          fill="#1a1a1a"
          stroke="#0a0a0a"
          strokeWidth="2"
        />

        {/* Neck opening */}
        <ellipse cx="400" cy="180" rx="60" ry="40" fill="#FFFFFF" opacity="0.1" />
      </g>

      {/* Elegant Flamingo Logo - Championship Gold - chest placement */}
      <g id="logo" transform="translate(320, 320)">
        <path
          d="M 80 20 Q 85 25 90 35 Q 90 45 85 50 L 80 55 Q 75 57 70 60 L 65 65 Q 60 70 60 80 L 60 95 Q 60 105 65 110 L 70 115 Q 75 120 80 120 Q 82 120 85 122 L 87 130 Q 87 135 85 140 L 82 145 Q 80 150 75 150 Q 70 150 67 145 L 65 130 Q 65 120 70 115 L 72 110 Q 67 105 65 100 L 60 90 Q 55 80 55 70 L 55 60 Q 55 50 60 45 L 65 40 Q 70 35 75 32 L 80 30 Q 75 25 70 22 L 65 20 Q 60 17 55 20 L 50 25 Q 45 30 45 35 L 45 40 Q 50 40 55 42 L 60 45 Q 55 45 50 42 L 45 40 Q 40 35 40 30 L 40 25 Q 40 20 45 15 L 50 10 Q 55 7 60 7 L 70 10 Q 75 12 80 20 Z"
          fill="#d4af37"
          stroke="#d4af37"
          strokeWidth="1"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>

      {/* Subtle shadow for depth */}
      <ellipse cx="400" cy="850" rx="200" ry="30" fill="#000000" opacity="0.1" />

      {/* Product details text */}
      <text
        x="400"
        y="930"
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
        y="960"
        fontFamily="Helvetica, sans-serif"
        fontSize="16"
        fill="#666666"
        textAnchor="middle"
      >
        Premium Carbon Black T-Shirt
      </text>
    </svg>
  );
};

export default BlackTShirtMockup;
