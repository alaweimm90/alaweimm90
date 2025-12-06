import * as React from "react";
import { cn } from "@/lib/utils";

export interface LazyImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  src: string;
  alt: string;
  fallback?: React.ReactNode;
  priority?: boolean;
  srcSet?: string;
  sizes?: string;
}

const LazyImage = React.forwardRef<HTMLImageElement, LazyImageProps>(
  ({ className, src, srcSet, sizes, alt, fallback, priority = false, ...props }, ref) => {
    const [isLoaded, setIsLoaded] = React.useState(false);
    const [hasError, setHasError] = React.useState(false);

    const handleLoad = () => {
      setIsLoaded(true);
    };

    const handleError = () => {
      setHasError(true);
    };

    if (hasError && fallback) {
      return <>{fallback}</>;
    }

    return (
      <img
        ref={ref}
        src={src}
        srcSet={srcSet}
        sizes={sizes || "(max-width: 768px) 100vw, 50vw"}
        alt={alt}
        loading={priority ? "eager" : "lazy"}
        decoding={priority ? "sync" : "async"}
        fetchPriority={priority ? "high" : "auto"}
        className={cn(
          "transition-opacity duration-300",
          isLoaded ? "opacity-100" : "opacity-0",
          className
        )}
        onLoad={handleLoad}
        onError={handleError}
        {...props}
      />
    );
  }
);

LazyImage.displayName = "LazyImage";

export { LazyImage };
