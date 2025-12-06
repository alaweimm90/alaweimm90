interface IconicLogoProps {
  className?: string;
}

const IconicLogo = ({ className = 'w-8 h-8' }: IconicLogoProps) => {
  return (
    <div className={`${className} relative flex items-center justify-center`}>
      <svg
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full"
      >
        {/* Stylized "I" with supercar-inspired design */}
        <path d="M16 4L24 8V12L16 16L8 12V8L16 4Z" fill="currentColor" className="text-accent" />
        <rect x="14" y="12" width="4" height="12" fill="currentColor" className="text-foreground" />
        <path d="M8 24H24L22 28H10L8 24Z" fill="currentColor" className="text-accent" />
        {/* Accent lines for movement */}
        <rect
          x="6"
          y="14"
          width="2"
          height="1"
          fill="currentColor"
          className="text-accent opacity-60"
        />
        <rect
          x="24"
          y="16"
          width="2"
          height="1"
          fill="currentColor"
          className="text-accent opacity-60"
        />
        <rect
          x="4"
          y="18"
          width="3"
          height="1"
          fill="currentColor"
          className="text-accent opacity-40"
        />
        <rect
          x="25"
          y="20"
          width="3"
          height="1"
          fill="currentColor"
          className="text-accent opacity-40"
        />
      </svg>
    </div>
  );
};

export default IconicLogo;
