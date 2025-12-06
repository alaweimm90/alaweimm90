import { cn } from '@/lib/utils';

interface LuxurySerifProps {
  className?: string;
}

const LuxurySerif = ({ className }: LuxurySerifProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="180"
        height="50"
        viewBox="0 0 180 50"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500"
      >
        {/* LIVE in elegant serif */}
        <text
          x="10"
          y="32"
          fill="hsl(var(--lii-gold))"
          fontSize="28"
          fontWeight="300"
          fontFamily="serif"
          letterSpacing="3"
        >
          LIVE
        </text>

        {/* Decorative dot */}
        <circle cx="75" cy="28" r="2" fill="hsl(var(--lii-gold))" />

        {/* IT in serif */}
        <text
          x="85"
          y="32"
          fill="hsl(var(--lii-gold))"
          fontSize="24"
          fontWeight="300"
          fontFamily="serif"
          letterSpacing="2"
        >
          IT
        </text>

        {/* Decorative dot */}
        <circle cx="115" cy="28" r="2" fill="hsl(var(--lii-gold))" />

        {/* ICONIC in serif */}
        <text
          x="125"
          y="32"
          fill="hsl(var(--lii-gold))"
          fontSize="28"
          fontWeight="300"
          fontFamily="serif"
          letterSpacing="3"
        >
          ICONIC
        </text>

        {/* Top flourish */}
        <path
          d="M10 8 Q90 5, 170 8"
          stroke="hsl(var(--lii-champagne))"
          strokeWidth="0.5"
          opacity="0.4"
          fill="none"
        />

        {/* Bottom flourish */}
        <path
          d="M10 42 Q90 45, 170 42"
          stroke="hsl(var(--lii-champagne))"
          strokeWidth="0.5"
          opacity="0.4"
          fill="none"
        />
      </svg>
    </div>
  );
};

export default LuxurySerif;
