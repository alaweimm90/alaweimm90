import { cn } from '@/lib/utils';

interface ScriptEleganceProps {
  className?: string;
}

const ScriptElegance = ({ className }: ScriptEleganceProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="160"
        height="50"
        viewBox="0 0 160 50"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500"
      >
        {/* Flowing L */}
        <path
          d="M10 10 Q10 30, 10 35 L25 35"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
        />
        {/* Elegant I */}
        <path
          d="M35 15 L35 35"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <circle cx="35" cy="10" r="1.5" fill="hsl(var(--lii-gold))" />
        {/* Flowing V */}
        <path
          d="M45 15 Q50 30, 55 35 Q60 30, 65 15"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
        />
        {/* Elegant E */}
        <path
          d="M75 15 L75 35 M75 15 L90 15 M75 25 L87 25 M75 35 L90 35"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
        />

        {/* Decorative swash */}
        <path
          d="M10 40 Q80 42, 150 40"
          stroke="hsl(var(--lii-champagne))"
          strokeWidth="0.5"
          opacity="0.5"
          fill="none"
        />

        <text
          x="95"
          y="32"
          fill="hsl(var(--lii-gold))"
          fontSize="14"
          fontWeight="300"
          fontFamily="serif"
          fontStyle="italic"
        >
          it iconic
        </text>
      </svg>
    </div>
  );
};

export default ScriptElegance;
