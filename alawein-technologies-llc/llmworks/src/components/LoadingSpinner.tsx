import { cn } from "@/lib/utils";
import { Swords, Zap, BarChart3 } from "lucide-react";
import { useState, useEffect } from "react";

interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg";
  className?: string;
  text?: string;
  variant?: "default" | "battle" | "pulse" | "dots";
}

const BattleLoadingAnimation = ({ size = "md" }: { size: string }) => {
  const [currentIcon, setCurrentIcon] = useState(0);
  const icons = [Swords, Zap, BarChart3];
  
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIcon((prev) => (prev + 1) % icons.length);
    }, 800);
    return () => clearInterval(interval);
  }, []);

  const sizeClasses = {
    sm: "h-6 w-6",
    md: "h-10 w-10",
    lg: "h-16 w-16"
  };

  const CurrentIcon = icons[currentIcon];

  return (
    <div className="glass-panel p-4 rounded-2xl relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-secondary/5 animate-pulse"></div>
      <CurrentIcon 
        className={cn(
          "text-primary transition-all duration-500 animate-pulse relative z-10",
          sizeClasses[size]
        )}
      />
    </div>
  );
};

const PulseLoadingAnimation = ({ size = "md" }: { size: string }) => {
  const sizeClasses = {
    sm: "h-4 w-4",
    md: "h-8 w-8",
    lg: "h-12 w-12"
  };

  return (
    <div className="relative">
      {/* Outer pulse ring */}
      <div className={cn(
        "absolute inset-0 rounded-full border-2 border-primary/30 animate-ping",
        sizeClasses[size]
      )} />
      {/* Middle pulse ring */}
      <div 
        className={cn(
          "absolute inset-1 rounded-full border border-primary/50 animate-pulse",
          size === "lg" ? "inset-2" : size === "md" ? "inset-1" : "inset-0.5"
        )}
        style={{ animationDelay: "0.2s" }}
      />
      {/* Inner core */}
      <div className={cn(
        "rounded-full bg-gradient-to-br from-primary to-secondary animate-spin",
        sizeClasses[size]
      )} />
    </div>
  );
};

const DotsLoadingAnimation = () => {
  return (
    <div className="flex items-center gap-2">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="w-2 h-2 rounded-full bg-primary animate-bounce"
          style={{ 
            animationDelay: `${i * 0.2}s`,
            animationDuration: "1s"
          }}
        />
      ))}
    </div>
  );
};

export const LoadingSpinner = ({ 
  size = "md", 
  className, 
  text,
  variant = "default"
}: LoadingSpinnerProps) => {
  const sizeClasses = {
    sm: "h-4 w-4",
    md: "h-8 w-8", 
    lg: "h-12 w-12"
  };

  const renderLoader = () => {
    switch (variant) {
      case "battle":
        return <BattleLoadingAnimation size={size} />;
      case "pulse":
        return <PulseLoadingAnimation size={size} />;
      case "dots":
        return <DotsLoadingAnimation />;
      default:
        return (
          <div 
            className={cn(
              "animate-spin rounded-full border-2 border-muted/20 border-t-primary shadow-lg",
              sizeClasses[size]
            )}
            aria-hidden="true"
          />
        );
    }
  };

  return (
    <div 
      className={cn("flex flex-col items-center gap-4 stagger-children", className)} 
      role="status" 
      aria-live="polite"
    >
      <div style={{'--stagger-index': 0} as React.CSSProperties}>
        {renderLoader()}
      </div>
      {text && (
        <div style={{'--stagger-index': 1} as React.CSSProperties}>
          <span className="body-elegant text-sm text-muted-foreground animate-pulse">
            {text}
          </span>
        </div>
      )}
      <span className="sr-only">Loading...</span>
    </div>
  );
};

export const PageLoader = ({ text = "Preparing combat systems..." }: { text?: string }) => (
  <div 
    className="min-h-screen flex items-center justify-center relative overflow-hidden"
    style={{
      background: `
        radial-gradient(circle at 25% 25%, hsl(var(--color-primary) / 0.03), transparent 50%),
        radial-gradient(circle at 75% 75%, hsl(var(--color-secondary) / 0.02), transparent 50%),
        var(--color-background)
      `
    }}
  >
    {/* Sophisticated Background Effects */}
    <div className="absolute inset-0 subtle-texture opacity-20"></div>
    <div className="absolute top-1/4 left-1/4 w-px h-32 bg-gradient-to-b from-primary/20 to-transparent"></div>
    <div className="absolute bottom-1/4 right-1/3 w-px h-24 bg-gradient-to-t from-secondary/15 to-transparent"></div>
    
    <div className="relative z-10 text-center">
      <LoadingSpinner size="lg" text={text} variant="battle" />
    </div>
  </div>
);

export const InlineLoader = ({ 
  text = "Loading...", 
  size = "sm" 
}: { 
  text?: string;
  size?: "sm" | "md" | "lg";
}) => (
  <div className="flex items-center gap-3 glass-panel p-3 rounded-lg">
    <LoadingSpinner size={size} variant="pulse" />
    <span className="body-elegant text-sm text-muted-foreground">
      {text}
    </span>
  </div>
);

export const SkeletonLoader = ({ 
  className,
  variant = "rectangular"
}: { 
  className?: string;
  variant?: "rectangular" | "circular" | "text";
}) => {
  const baseClasses = "animate-pulse bg-gradient-to-r from-muted/50 to-muted/20";
  
  const variantClasses = {
    rectangular: "rounded-lg",
    circular: "rounded-full aspect-square",
    text: "rounded-md h-4"
  };

  return (
    <div 
      className={cn(
        baseClasses,
        variantClasses[variant],
        className
      )}
      aria-hidden="true"
    />
  );
};