import { cn } from '@/lib/utils';

interface DiagonalDynamicProps {
  className?: string;
}

const DiagonalDynamic = ({ className }: DiagonalDynamicProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="160"
        height="80"
        viewBox="0 0 160 80"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500"
      >
        {/* Diagonal slant background */}
        <path d="M10 70 L40 10 L150 10 L120 70 Z" fill="hsl(var(--lii-gold))" opacity="0.1" />

        {/* LIVE */}
        <text
          x="20"
          y="50"
          fill="hsl(var(--lii-gold))"
          fontSize="24"
          fontWeight="700"
          fontFamily="sans-serif"
          transform="skewX(-10)"
          letterSpacing="2"
        >
          LIVE
        </text>

        {/* IT */}
        <text
          x="75"
          y="50"
          fill="hsl(var(--lii-gold))"
          fontSize="20"
          fontWeight="400"
          fontFamily="sans-serif"
          transform="skewX(-10)"
        >
          IT
        </text>

        {/* ICONIC */}
        <text
          x="100"
          y="50"
          fill="hsl(var(--lii-gold))"
          fontSize="24"
          fontWeight="700"
          fontFamily="sans-serif"
          transform="skewX(-10)"
          letterSpacing="2"
        >
          ICONIC
        </text>

        {/* Speed lines */}
        <path d="M5 25 L15 25" stroke="hsl(var(--lii-gold))" strokeWidth="1" opacity="0.3" />
        <path d="M5 35 L20 35" stroke="hsl(var(--lii-gold))" strokeWidth="1" opacity="0.3" />
        <path d="M5 45 L18 45" stroke="hsl(var(--lii-gold))" strokeWidth="1" opacity="0.3" />
      </svg>
    </div>
  );
};

export default DiagonalDynamic;
