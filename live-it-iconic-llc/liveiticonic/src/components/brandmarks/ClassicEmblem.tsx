import { cn } from '@/lib/utils';

interface ClassicEmblemProps {
  className?: string;
}

const ClassicEmblem = ({ className }: ClassicEmblemProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="120"
        height="120"
        viewBox="0 0 120 120"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-700 group-hover:scale-105"
      >
        {/* Outer ornate border */}
        <circle cx="60" cy="60" r="55" stroke="hsl(var(--lii-gold))" strokeWidth="2" fill="none" />
        <circle
          cx="60"
          cy="60"
          r="50"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="0.5"
          fill="none"
          opacity="0.5"
        />

        {/* Decorative corners */}
        <circle cx="60" cy="10" r="2" fill="hsl(var(--lii-gold))" />
        <circle cx="60" cy="110" r="2" fill="hsl(var(--lii-gold))" />
        <circle cx="10" cy="60" r="2" fill="hsl(var(--lii-gold))" />
        <circle cx="110" cy="60" r="2" fill="hsl(var(--lii-gold))" />

        {/* Top text: LIVE */}
        <text
          x="60"
          y="40"
          fill="hsl(var(--lii-gold))"
          fontSize="16"
          fontWeight="300"
          fontFamily="serif"
          textAnchor="middle"
          letterSpacing="3"
        >
          LIVE
        </text>

        {/* Center initials: L.I.I */}
        <text
          x="60"
          y="68"
          fill="hsl(var(--lii-gold))"
          fontSize="24"
          fontWeight="600"
          fontFamily="serif"
          textAnchor="middle"
        >
          L.I.I
        </text>

        {/* Bottom text: ICONIC */}
        <text
          x="60"
          y="88"
          fill="hsl(var(--lii-gold))"
          fontSize="16"
          fontWeight="300"
          fontFamily="serif"
          textAnchor="middle"
          letterSpacing="3"
        >
          ICONIC
        </text>

        {/* Laurel wreaths */}
        <path
          d="M30 95 Q40 90, 50 95"
          stroke="hsl(var(--lii-champagne))"
          strokeWidth="1"
          fill="none"
          opacity="0.6"
        />
        <path
          d="M70 95 Q80 90, 90 95"
          stroke="hsl(var(--lii-champagne))"
          strokeWidth="1"
          fill="none"
          opacity="0.6"
        />
      </svg>
    </div>
  );
};

export default ClassicEmblem;
