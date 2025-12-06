import { cn } from '@/lib/utils';

interface ArtDecoStyleProps {
  className?: string;
}

const ArtDecoStyle = ({ className }: ArtDecoStyleProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="180"
        height="60"
        viewBox="0 0 180 60"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500"
      >
        {/* Art Deco geometric frame */}
        <path
          d="M10 10 L170 10 L175 15 L175 45 L170 50 L10 50 L5 45 L5 15 Z"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          fill="none"
        />

        {/* Inner geometric details */}
        <path
          d="M15 15 L165 15 M15 45 L165 45"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="0.5"
          opacity="0.4"
        />

        {/* LIVE in Art Deco style */}
        <text
          x="25"
          y="37"
          fill="hsl(var(--lii-gold))"
          fontSize="22"
          fontWeight="600"
          fontFamily="sans-serif"
          letterSpacing="4"
        >
          LIVE
        </text>

        {/* Vertical separator */}
        <path d="M75 20 L75 40" stroke="hsl(var(--lii-gold))" strokeWidth="2" />

        {/* IT */}
        <text
          x="85"
          y="37"
          fill="hsl(var(--lii-gold))"
          fontSize="18"
          fontWeight="400"
          fontFamily="sans-serif"
          letterSpacing="2"
        >
          IT
        </text>

        {/* Vertical separator */}
        <path d="M110 20 L110 40" stroke="hsl(var(--lii-gold))" strokeWidth="2" />

        {/* ICONIC */}
        <text
          x="120"
          y="37"
          fill="hsl(var(--lii-gold))"
          fontSize="22"
          fontWeight="600"
          fontFamily="sans-serif"
          letterSpacing="4"
        >
          ICONIC
        </text>

        {/* Corner accents */}
        <rect x="8" y="8" width="4" height="4" fill="hsl(var(--lii-gold))" />
        <rect x="168" y="8" width="4" height="4" fill="hsl(var(--lii-gold))" />
        <rect x="8" y="48" width="4" height="4" fill="hsl(var(--lii-gold))" />
        <rect x="168" y="48" width="4" height="4" fill="hsl(var(--lii-gold))" />
      </svg>
    </div>
  );
};

export default ArtDecoStyle;
