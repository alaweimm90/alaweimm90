import { cn } from '@/lib/utils';

interface InitialsCircleProps {
  className?: string;
}

const InitialsCircle = ({ className }: InitialsCircleProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="100"
        height="100"
        viewBox="0 0 100 100"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-700 group-hover:rotate-6 group-hover:scale-105"
      >
        {/* Outer circle */}
        <circle cx="50" cy="50" r="45" stroke="hsl(var(--lii-gold))" strokeWidth="2" fill="none" />

        {/* Inner circle */}
        <circle
          cx="50"
          cy="50"
          r="38"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="0.5"
          fill="none"
          opacity="0.4"
        />

        {/* L.I.I centered */}
        <text
          x="50"
          y="60"
          fill="hsl(var(--lii-gold))"
          fontSize="32"
          fontWeight="300"
          fontFamily="serif"
          textAnchor="middle"
        >
          L.I.I
        </text>

        {/* Bottom arc text */}
        <path id="bottomArc" d="M 20,50 A 30,30 0 0,0 80,50" fill="none" />
        <text
          fill="hsl(var(--lii-champagne))"
          fontSize="8"
          fontWeight="400"
          fontFamily="sans-serif"
          letterSpacing="2"
        >
          <textPath href="#bottomArc" startOffset="50%" textAnchor="middle">
            LIVE IT ICONIC
          </textPath>
        </text>
      </svg>
    </div>
  );
};

export default InitialsCircle;
