import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface OptimizedImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  src: string;
  alt: string;
  fallback?: string;
  placeholder?: string;
  className?: string;
  priority?: boolean;
  blur?: boolean;
}

export const OptimizedImage: React.FC<OptimizedImageProps> = ({
  src,
  alt,
  fallback = '/placeholder-image.jpg',
  placeholder,
  className,
  priority = false,
  blur = true,
  ...props
}) => {
  const [isLoading, setIsLoading] = useState(true);
  const [isInView, setIsInView] = useState(priority);
  const [hasError, setHasError] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const [imageSrc, setImageSrc] = useState(priority ? src : '');

  // Intersection Observer for lazy loading
  useEffect(() => {
    if (priority || !imgRef.current) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          setImageSrc(src);
          observer.disconnect();
        }
      },
      {
        threshold: 0.1,
        rootMargin: '50px'
      }
    );

    observer.observe(imgRef.current);
    return () => observer.disconnect();
  }, [src, priority]);

  const handleLoad = () => {
    setIsLoading(false);
  };

  const handleError = () => {
    setHasError(true);
    setIsLoading(false);
    if (fallback) {
      setImageSrc(fallback);
    }
  };

  return (
    <div className={cn('relative overflow-hidden', className)}>
      {/* Placeholder while loading */}
      {isLoading && (
        <div 
          className={cn(
            'absolute inset-0 bg-gradient-to-r from-gray-200 via-gray-300 to-gray-200 animate-pulse',
            blur && 'backdrop-blur-sm'
          )}
          style={{
            backgroundImage: placeholder ? `url(${placeholder})` : undefined,
            backgroundSize: 'cover',
            backgroundPosition: 'center'
          }}
        />
      )}
      
      {/* Main image */}
      <img
        ref={imgRef}
        src={imageSrc}
        alt={alt}
        className={cn(
          'transition-opacity duration-500',
          isLoading ? 'opacity-0' : 'opacity-100',
          className
        )}
        onLoad={handleLoad}
        onError={handleError}
        loading={priority ? 'eager' : 'lazy'}
        decoding="async"
        {...props}
      />
      
      {/* Error state */}
      {hasError && !fallback && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 text-gray-500">
          <span className="text-sm">Failed to load image</span>
        </div>
      )}
    </div>
  );
};