import { cn } from '@/lib/utils';

interface ModernWordmarkProps {
  className?: string;
}

const ModernWordmark = ({ className }: ModernWordmarkProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="180"
        height="40"
        viewBox="0 0 180 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500"
      >
        {/* L */}
        <path
          d="M10 8 L10 32 L20 32"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
        {/* I */}
        <rect x="30" y="8" width="3" height="24" fill="hsl(var(--lii-gold))" />
        {/* V */}
        <path
          d="M40 8 L48 32 L56 8"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
        {/* E */}
        <path
          d="M66 8 L66 32 M66 8 L76 8 M66 20 L74 20 M66 32 L76 32"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2.5"
          strokeLinecap="round"
        />

        {/* IT */}
        <text
          x="90"
          y="28"
          fill="hsl(var(--lii-gold))"
          fontSize="20"
          fontWeight="300"
          fontFamily="sans-serif"
        >
          IT
        </text>

        {/* ICONIC - Stylized */}
        <text
          x="115"
          y="28"
          fill="hsl(var(--lii-gold))"
          fontSize="20"
          fontWeight="600"
          fontFamily="sans-serif"
          letterSpacing="2"
        >
          ICONIC
        </text>

        {/* Underline accent */}
        <line
          x1="10"
          y1="36"
          x2="175"
          y2="36"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="0.5"
          opacity="0.4"
        />
      </svg>
    </div>
  );
};

export default ModernWordmark;
