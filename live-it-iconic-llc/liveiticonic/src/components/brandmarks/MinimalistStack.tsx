import { cn } from '@/lib/utils';

interface MinimalistStackProps {
  className?: string;
}

const MinimalistStack = ({ className }: MinimalistStackProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="120"
        height="60"
        viewBox="0 0 120 60"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500"
      >
        {/* LIVE stacked */}
        <text
          x="10"
          y="18"
          fill="hsl(var(--lii-gold))"
          fontSize="16"
          fontWeight="200"
          fontFamily="sans-serif"
          letterSpacing="4"
        >
          LIVE
        </text>
        {/* IT stacked */}
        <text
          x="20"
          y="34"
          fill="hsl(var(--lii-gold))"
          fontSize="16"
          fontWeight="300"
          fontFamily="sans-serif"
          letterSpacing="6"
        >
          IT
        </text>
        {/* ICONIC stacked */}
        <text
          x="10"
          y="50"
          fill="hsl(var(--lii-gold))"
          fontSize="16"
          fontWeight="600"
          fontFamily="sans-serif"
          letterSpacing="3"
        >
          ICONIC
        </text>

        {/* Side accent bar */}
        <rect x="2" y="8" width="2" height="44" fill="hsl(var(--lii-gold))" opacity="0.6" />
      </svg>
    </div>
  );
};

export default MinimalistStack;
