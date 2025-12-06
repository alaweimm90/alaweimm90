import React from 'react';

interface AISecurityIconProps {
  className?: string;
  size?: number;
}

export default function AISecurityIcon({
  className = "",
  size = 24,
}: AISecurityIconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      role="img"
      aria-label="AI security and attribution detection robot mascot"
    >
      <title>AI security and attribution detection robot mascot</title>

      {/* Minimal, centered happy robot head */}

      {/* Antenna */}
      <line x1="12" y1="4.8" x2="12" y2="6.4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="12" cy="4.2" r="1" fill="currentColor" />

      {/* Ears */}
      <rect x="5.5" y="8.2" width="1.6" height="3.6" rx="0.6" fill="currentColor" opacity="0.4" />
      <rect x="16.9" y="8.2" width="1.6" height="3.6" rx="0.6" fill="currentColor" opacity="0.4" />

      {/* Head */}
      <rect
        x="6.2"
        y="6.5"
        width="11.6"
        height="8.2"
        rx="2.6"
        fill="currentColor"
        opacity="0.06"
      />
      <rect
        x="6.2"
        y="6.5"
        width="11.6"
        height="8.2"
        rx="2.6"
        stroke="currentColor"
        strokeWidth="1.5"
        fill="none"
      />

      {/* Eyes */}
      <circle cx="9.4" cy="9.5" r="1" fill="currentColor" />
      <circle cx="14.6" cy="9.5" r="1" fill="currentColor" />

      {/* Smile */}
      <path d="M8.8 11.6 Q12 13.4 15.2 11.6" stroke="currentColor" strokeWidth="1.6" fill="none" strokeLinecap="round" />

      {/* Neck / base */}
      <rect x="9.2" y="14.9" width="5.6" height="1.9" rx="0.9" fill="currentColor" opacity="0.18" />
    </svg>
  );
}
