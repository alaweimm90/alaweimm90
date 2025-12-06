import { cn } from '@/lib/utils';

interface HandwrittenSignatureProps {
  className?: string;
}

const HandwrittenSignature = ({ className }: HandwrittenSignatureProps) => {
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
        {/* Flowing handwritten L */}
        <path
          d="M15 15 Q15 35, 15 40 Q15 45, 30 45"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2.5"
          strokeLinecap="round"
          fill="none"
        />

        {/* Handwritten i */}
        <path
          d="M40 25 L40 45"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <circle cx="40" cy="18" r="1.5" fill="hsl(var(--lii-gold))" />

        {/* Handwritten v */}
        <path
          d="M50 25 Q55 40, 60 45 Q65 40, 70 25"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
        />

        {/* Handwritten e */}
        <path
          d="M80 35 Q90 25, 95 35 Q95 45, 80 45"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
        />

        {/* it - smaller */}
        <text
          x="105"
          y="42"
          fill="hsl(var(--lii-gold))"
          fontSize="16"
          fontWeight="300"
          fontFamily="cursive"
          fontStyle="italic"
        >
          it
        </text>

        {/* Iconic - cursive */}
        <text
          x="125"
          y="42"
          fill="hsl(var(--lii-gold))"
          fontSize="20"
          fontWeight="400"
          fontFamily="cursive"
          fontStyle="italic"
        >
          Iconic
        </text>

        {/* Underline flourish */}
        <path
          d="M15 50 Q90 55, 170 48"
          stroke="hsl(var(--lii-champagne))"
          strokeWidth="1"
          opacity="0.5"
          fill="none"
        />
      </svg>
    </div>
  );
};

export default HandwrittenSignature;
