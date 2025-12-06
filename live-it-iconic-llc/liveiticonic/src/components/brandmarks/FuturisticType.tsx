import { cn } from '@/lib/utils';

interface FuturisticTypeProps {
  className?: string;
}

const FuturisticType = ({ className }: FuturisticTypeProps) => {
  return (
    <div className={cn('relative group', className)}>
      <svg
        width="200"
        height="50"
        viewBox="0 0 200 50"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-500"
      >
        {/* Futuristic L */}
        <path
          d="M10 10 L10 35 L25 35 M10 10 L15 8"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="3"
          strokeLinecap="round"
        />

        {/* Futuristic I */}
        <path
          d="M35 10 L35 35 M33 10 L37 10 M33 35 L37 35"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="3"
          strokeLinecap="round"
        />

        {/* Futuristic V */}
        <path
          d="M45 10 L52 35 M59 10 L52 35"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="3"
          strokeLinecap="round"
        />

        {/* Futuristic E */}
        <path
          d="M69 10 L69 35 M69 10 L82 10 M69 22.5 L79 22.5 M69 35 L82 35"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="3"
          strokeLinecap="round"
        />

        {/* IT - lighter weight */}
        <path
          d="M92 15 L92 35 M105 15 L105 35 M100 15 L110 15"
          stroke="hsl(var(--lii-gold))"
          strokeWidth="2"
          strokeLinecap="round"
        />

        {/* ICONIC - bold */}
        <text
          x="120"
          y="32"
          fill="hsl(var(--lii-gold))"
          fontSize="22"
          fontWeight="800"
          fontFamily="sans-serif"
          letterSpacing="3"
        >
          ICONIC
        </text>

        {/* Tech accents */}
        <rect x="8" y="5" width="2" height="2" fill="hsl(var(--lii-gold))" opacity="0.5" />
        <rect x="195" y="5" width="2" height="2" fill="hsl(var(--lii-gold))" opacity="0.5" />
        <rect x="8" y="43" width="2" height="2" fill="hsl(var(--lii-gold))" opacity="0.5" />
        <rect x="195" y="43" width="2" height="2" fill="hsl(var(--lii-gold))" opacity="0.5" />
      </svg>
    </div>
  );
};

export default FuturisticType;
