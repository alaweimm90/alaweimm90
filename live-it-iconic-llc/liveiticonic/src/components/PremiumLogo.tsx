import { cn } from '@/lib/utils';

interface PremiumLogoProps {
  className?: string;
}

const PremiumLogo = ({ className }: PremiumLogoProps) => {
  return (
    <div className={cn('relative group', className)}>
      {/* Main Logo Mark */}
      <svg
        width="40"
        height="40"
        viewBox="0 0 40 40"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="transition-all duration-700 group-hover:scale-105"
      >
        {/* Outer Ring - Gold */}
        <circle
          cx="20"
          cy="20"
          r="18"
          stroke="url(#goldGradient)"
          strokeWidth="1.5"
          fill="none"
          className="group-hover:stroke-[2]"
        />

        {/* Inner Geometric Pattern */}
        <path
          d="M12 20 L20 12 L28 20 L20 28 Z"
          stroke="url(#champagneGradient)"
          strokeWidth="1"
          fill="url(#subtleFill)"
          className="group-hover:fill-opacity-30"
        />

        {/* Center Diamond */}
        <circle cx="20" cy="20" r="3" fill="url(#goldGradient)" className="group-hover:r-[3.5]" />

        {/* Accent Lines */}
        <line
          x1="8"
          y1="20"
          x2="12"
          y2="20"
          stroke="url(#bronzeGradient)"
          strokeWidth="1.5"
          opacity="0.6"
        />
        <line
          x1="28"
          y1="20"
          x2="32"
          y2="20"
          stroke="url(#bronzeGradient)"
          strokeWidth="1.5"
          opacity="0.6"
        />
        <line
          x1="20"
          y1="8"
          x2="20"
          y2="12"
          stroke="url(#bronzeGradient)"
          strokeWidth="1.5"
          opacity="0.6"
        />
        <line
          x1="20"
          y1="28"
          x2="20"
          y2="32"
          stroke="url(#bronzeGradient)"
          strokeWidth="1.5"
          opacity="0.6"
        />

        <defs>
          <linearGradient id="goldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="hsl(42, 55%, 65%)" />
            <stop offset="100%" stopColor="hsl(42, 55%, 45%)" />
          </linearGradient>

          <linearGradient id="champagneGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="hsl(45, 35%, 85%)" />
            <stop offset="100%" stopColor="hsl(45, 35%, 75%)" />
          </linearGradient>

          <linearGradient id="bronzeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="hsl(28, 40%, 50%)" />
            <stop offset="100%" stopColor="hsl(28, 40%, 30%)" />
          </linearGradient>

          <radialGradient id="subtleFill" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="hsl(42, 55%, 55%)" stopOpacity="0.1" />
            <stop offset="100%" stopColor="hsl(42, 55%, 55%)" stopOpacity="0.05" />
          </radialGradient>
        </defs>
      </svg>

      {/* Subtle Glow Effect */}
      <div className="absolute inset-0 bg-lii-gold/10 rounded-full scale-0 group-hover:scale-125 transition-all duration-700 blur-xl opacity-0 group-hover:opacity-60 -z-10"></div>
    </div>
  );
};

export default PremiumLogo;
