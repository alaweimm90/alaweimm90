import { cn } from '@/lib/utils';

interface MonogramShieldProps {
  className?: string;
}

const MonogramShield = ({ className }: MonogramShieldProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="80"
        height="90"
        viewBox="0 0 80 90"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500 group-hover:scale-105"
      >
        {/* Shield shape */}
        <path
          d="M40 5 L70 20 L70 50 Q70 70, 40 85 Q10 70, 10 50 L10 20 Z"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          fill="none"
        />

        {/* Inner shield detail */}
        <path
          d="M40 10 L65 23 L65 50 Q65 68, 40 80 Q15 68, 15 50 L15 23 Z"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="0.5"
          fill="none"
          opacity="0.3"
        />

        {/* Monogram L.I.I */}
        <text
          x="40"
          y="50"
          fill="hsl(var(--lii-gold))"
          fontSize="20"
          fontWeight="600"
          fontFamily="serif"
          textAnchor="middle"
        >
          LII
        </text>

        {/* Crown accent */}
        <path
          d="M30 18 L35 15 L40 18 L45 15 L50 18"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Bottom banner */}
        <path
          d="M25 65 L40 70 L55 65"
          stroke="hsl(var(--lii-champagne))"
          strokeWidth="1"
          opacity="0.6"
        />
      </svg>
    </div>
  );
};

export default MonogramShield;
