/**
 * Black Cap Product Mockup
 * Features: Tropical Tanager logo in 3D gold embroidery
 * Structured fit, premium construction for luxury lifestyle
 */

import React from 'react';

interface ProductMockupProps {
  className?: string;
  width?: number;
  height?: number;
}

export const BlackCapMockup: React.FC<ProductMockupProps> = ({
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

      {/* Cap front view - carbon black */}
      <g id="cap">
        {/* Visor/Bill */}
        <ellipse
          cx="400"
          cy="520"
          rx="280"
          ry="80"
          fill="#0a0a0a"
          stroke="#000000"
          strokeWidth="3"
        />
        <ellipse
          cx="400"
          cy="510"
          rx="270"
          ry="70"
          fill="#1a1a1a"
          stroke="#0a0a0a"
          strokeWidth="2"
        />

        {/* Cap crown - 6 panels */}
        <g id="crown">
          {/* Center front panel */}
          <path
            d="M 350 300 L 320 480 Q 340 500 400 500 Q 460 500 480 480 L 450 300 Q 430 250 400 250 Q 370 250 350 300 Z"
            fill="#1a1a1a"
            stroke="#0a0a0a"
            strokeWidth="2"
          />
          {/* Left panel */}
          <path
            d="M 350 300 L 320 480 L 250 470 L 280 290 Q 310 260 350 300 Z"
            fill="#151515"
            stroke="#0a0a0a"
            strokeWidth="2"
          />
          {/* Right panel */}
          <path
            d="M 450 300 L 480 480 L 550 470 L 520 290 Q 490 260 450 300 Z"
            fill="#151515"
            stroke="#0a0a0a"
            strokeWidth="2"
          />
          {/* Far left panel */}
          <path
            d="M 280 290 L 250 470 L 200 450 L 220 280 Q 240 265 280 290 Z"
            fill="#0f0f0f"
            stroke="#0a0a0a"
            strokeWidth="2"
          />
          {/* Far right panel */}
          <path
            d="M 520 290 L 550 470 L 600 450 L 580 280 Q 560 265 520 290 Z"
            fill="#0f0f0f"
            stroke="#0a0a0a"
            strokeWidth="2"
          />

          {/* Top button */}
          <circle cx="400" cy="250" r="12" fill="#d4af37" stroke="#b8941f" strokeWidth="2" />
        </g>

        {/* Panel stitching details */}
        <line x1="350" y1="300" x2="330" y2="480" stroke="#0a0a0a" strokeWidth="1.5" />
        <line x1="450" y1="300" x2="470" y2="480" stroke="#0a0a0a" strokeWidth="1.5" />
        <line x1="280" y1="290" x2="260" y2="470" stroke="#0a0a0a" strokeWidth="1.5" />
        <line x1="520" y1="290" x2="540" y2="470" stroke="#0a0a0a" strokeWidth="1.5" />

        {/* Eyelets for ventilation */}
        <circle cx="300" cy="400" r="6" fill="#0a0a0a" stroke="#d4af37" strokeWidth="1" />
        <circle cx="500" cy="400" r="6" fill="#0a0a0a" stroke="#d4af37" strokeWidth="1" />
      </g>

      {/* Tropical Tanager Logo - Championship Gold - center front panel */}
      <g id="logo" transform="translate(340, 330)">
        <path
          d="M 70 30 Q 65 30 60 35 L 55 40 Q 50 45 50 50 L 50 55 Q 50 60 55 65 L 60 70 Q 65 75 70 75 L 75 75 Q 80 75 85 70 L 90 65 Q 95 60 95 55 L 95 50 Q 95 45 90 40 L 85 35 Q 80 30 75 30 L 70 30 Z M 75 75 L 80 80 Q 85 85 85 90 L 85 100 Q 85 110 80 115 L 75 120 Q 70 125 65 125 L 60 125 Q 55 125 50 120 L 45 115 Q 40 110 40 100 L 40 90 Q 40 85 45 80 L 50 75 L 75 75 Z M 65 125 L 60 130 Q 55 135 55 140 L 55 145 Q 55 150 60 155 L 65 160 Q 70 165 75 160 L 80 155 Q 85 150 85 145 L 85 140 Q 85 135 80 130 L 75 125 L 65 125 Z M 70 30 Q 75 25 80 20 L 85 15 Q 90 10 95 10 Q 100 10 105 15 L 110 20 Q 115 25 115 30 L 115 35 Q 115 40 110 45 L 105 50 Q 100 55 95 55 L 95 50 Q 100 45 105 40 L 110 35 Q 115 30 115 25 Q 115 20 110 15 L 105 10 Q 100 5 95 5 Q 90 5 85 10 L 80 15 Q 75 20 70 25 L 70 30 Z"
          fill="#d4af37"
          stroke="#d4af37"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* 3D embroidery effect */}
        <path
          d="M 70 30 Q 65 30 60 35 L 55 40 Q 50 45 50 50 L 50 55 Q 50 60 55 65 L 60 70 Q 65 75 70 75 L 75 75 Q 80 75 85 70 L 90 65 Q 95 60 95 55 L 95 50 Q 95 45 90 40 L 85 35 Q 80 30 75 30 L 70 30 Z"
          fill="none"
          stroke="#f0e68c"
          strokeWidth="0.5"
          strokeLinecap="round"
          opacity="0.6"
          transform="translate(-1, -1)"
        />
      </g>

      {/* Subtle shadow for depth */}
      <ellipse cx="400" cy="580" rx="300" ry="40" fill="#000000" opacity="0.1" />

      {/* Adjustable back strap hint */}
      <rect
        x="350"
        y="530"
        width="100"
        height="15"
        rx="3"
        fill="#1a1a1a"
        stroke="#0a0a0a"
        strokeWidth="1"
      />
      <circle cx="380" cy="537" r="3" fill="#d4af37" />
      <circle cx="400" cy="537" r="3" fill="#d4af37" />
      <circle cx="420" cy="537" r="3" fill="#d4af37" />

      {/* Product details text */}
      <text
        x="400"
        y="700"
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
        y="730"
        fontFamily="Helvetica, sans-serif"
        fontSize="16"
        fill="#666666"
        textAnchor="middle"
      >
        Premium Carbon Black Cap - Structured Fit
      </text>
    </svg>
  );
};

export default BlackCapMockup;
