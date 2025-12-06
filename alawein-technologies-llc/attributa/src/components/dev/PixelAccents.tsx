import React from "react";
import { cn } from "@/lib/utils";

interface PixelAccentsProps {
  variant?: "hero" | "cta" | "default";
  className?: string;
}

const Pixel: React.FC<{ className?: string }> = ({ className }) => (
  <div className={cn("w-2 h-2 md:w-3 md:h-3 rounded-sm", className)} />
);

export default function PixelAccents({ variant = "default", className }: PixelAccentsProps) {
  // Decorative, non-interactive layer
  return (
    <div
      aria-hidden
      className={cn(
        "pointer-events-none absolute inset-0 select-none",
        // Keep very subtle by default
        variant === "hero" && "opacity-60",
        variant === "cta" && "opacity-70",
        variant === "default" && "opacity-60",
        className
      )}
    >
      {/* Corner clusters - pixel motif */}
      <div className="absolute top-4 left-4 grid grid-cols-3 gap-1">
        <Pixel className="bg-primary/70" />
        <Pixel className="bg-primary/40" />
        <Pixel className="bg-accent/40" />
        <Pixel className="bg-primary/40" />
        <Pixel className="bg-accent/70" />
        <Pixel className="bg-primary/40" />
        <Pixel className="bg-accent/40" />
        <Pixel className="bg-primary/40" />
        <Pixel className="bg-accent/70" />
      </div>

      <div className="absolute top-6 right-6 grid grid-cols-2 gap-1">
        <Pixel className="bg-accent/70" />
        <Pixel className="bg-primary/40" />
        <Pixel className="bg-primary/70" />
        <Pixel className="bg-accent/40" />
      </div>

      <div className="absolute bottom-6 left-6 grid grid-cols-2 gap-1">
        <Pixel className="bg-primary/70" />
        <Pixel className="bg-accent/40" />
        <Pixel className="bg-accent/70" />
        <Pixel className="bg-primary/40" />
      </div>

      {/* Lightly drifting singles for subtle motion */}
      <div className="absolute left-1/3 top-8 pulse">
        <Pixel className="bg-primary/50" />
      </div>
      <div className="absolute left-2/3 top-1/3 pulse" style={{ animationDelay: "0.6s" } as React.CSSProperties}>
        <Pixel className="bg-accent/50" />
      </div>
      <div className="absolute right-1/4 bottom-1/4 pulse" style={{ animationDelay: "1s" } as React.CSSProperties}>
        <Pixel className="bg-primary/40" />
      </div>

      {/* Variant-specific accents */}
      {variant === "hero" && (
        <>
          <div className="absolute bottom-8 right-8 grid grid-cols-3 gap-1">
            <Pixel className="bg-primary/70" />
            <Pixel className="bg-accent/50" />
            <Pixel className="bg-primary/40" />
            <Pixel className="bg-accent/40" />
            <Pixel className="bg-primary/70" />
            <Pixel className="bg-accent/50" />
            <Pixel className="bg-primary/40" />
            <Pixel className="bg-accent/40" />
            <Pixel className="bg-primary/70" />
          </div>
        </>
      )}

      {variant === "cta" && (
        <>
          <div className="absolute inset-x-0 bottom-0 mx-auto mb-4 flex gap-1 justify-center">
            <Pixel className="bg-primary/60" />
            <Pixel className="bg-accent/60" />
            <Pixel className="bg-primary/60" />
            <Pixel className="bg-accent/60" />
            <Pixel className="bg-primary/60" />
          </div>
        </>
      )}
    </div>
  );
}
