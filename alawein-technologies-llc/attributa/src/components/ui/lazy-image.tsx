import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/lib/utils';

interface LazyImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  src: string;
  alt: string;
  className?: string;
  fallback?: React.ReactNode;
  placeholder?: string;
  priority?: boolean; /* For LCP images */
  sizes?: string; /* Responsive image sizes */
  srcSet?: string; /* Responsive image sources */
}

export function LazyImage({ 
  src, 
  alt, 
  className, 
  fallback,
  placeholder = "data:image/svg+xml,%3Csvg width='100' height='100' xmlns='http://www.w3.org/2000/svg'%3E%3Crect width='100%25' height='100%25' fill='%23f0f0f0'/%3E%3C/svg%3E",
  priority = false,
  sizes,
  srcSet,
  ...props 
}: LazyImageProps) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isError, setIsError] = useState(false);
  const [isInView, setIsInView] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { rootMargin: '50px' }
    );

    if (imgRef.current) {
      observer.observe(imgRef.current);
    }

    return () => observer.disconnect();
  }, []);

  if (isError && fallback) {
    return <>{fallback}</>;
  }

  return (
    <img
      ref={imgRef}
      src={priority ? src : (isInView ? src : placeholder)}
      alt={alt}
      className={cn(
        'transition-opacity duration-300',
        isLoaded ? 'opacity-100' : 'opacity-0',
        className
      )}
      onLoad={() => setIsLoaded(true)}
      onError={() => setIsError(true)}
      loading={priority ? "eager" : "lazy"}
      fetchpriority={priority ? "high" : undefined}
      decoding={priority ? "sync" : "async"}
      sizes={sizes}
      srcSet={srcSet}
      {...props}
    />
  );
}