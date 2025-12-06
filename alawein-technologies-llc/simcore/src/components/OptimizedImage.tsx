// React component for optimized responsive images

import React, { useRef, useEffect, useState } from 'react';
import { useImageOptimization } from '@/lib/image-optimization';
import { cn } from '@/lib/utils';

interface OptimizedImageProps {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  className?: string;
  loading?: 'lazy' | 'eager';
  priority?: boolean;
  quality?: number;
  sizes?: string;
  onLoad?: () => void;
  onError?: () => void;
  placeholder?: 'blur' | 'empty';
  blurDataURL?: string;
}

export const OptimizedImage: React.FC<OptimizedImageProps> = ({
  src,
  alt,
  width,
  height,
  className,
  loading = 'lazy',
  priority = false,
  quality = 85,
  sizes,
  onLoad,
  onError,
  placeholder = 'blur',
  blurDataURL,
}) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);
  const { optimizeImage, enableLazyLoading, optimalFormat } = useImageOptimization();

  const optimizedUrls = optimizeImage(src, { quality, format: optimalFormat });

  useEffect(() => {
    if (loading === 'lazy' && imgRef.current) {
      enableLazyLoading(imgRef.current);
    }
  }, [loading, enableLazyLoading]);

  // Preload critical images
  useEffect(() => {
    if (priority && src) {
      const link = document.createElement('link');
      link.rel = 'preload';
      link.href = optimizedUrls.src;
      link.as = 'image';
      if (optimalFormat === 'webp') link.type = 'image/webp';
      if (optimalFormat === 'avif') link.type = 'image/avif';
      document.head.appendChild(link);

      return () => {
        document.head.removeChild(link);
      };
    }
  }, [priority, src, optimizedUrls.src, optimalFormat]);

  const handleLoad = () => {
    setIsLoaded(true);
    onLoad?.();
  };

  const handleError = () => {
    setHasError(true);
    onError?.();
  };

  // Generate placeholder
  const placeholderSrc = blurDataURL || (
    placeholder === 'blur' 
      ? `data:image/svg+xml;base64,${btoa(`
          <svg width="${width || 400}" height="${height || 300}" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#f3f4f6;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#e5e7eb;stop-opacity:1" />
              </linearGradient>
            </defs>
            <rect width="100%" height="100%" fill="url(#grad)" />
            <text x="50%" y="50%" text-anchor="middle" dy=".3em" fill="#9ca3af" font-family="sans-serif" font-size="14">Loading...</text>
          </svg>
        `)}`
      : undefined
  );

  if (hasError) {
    return (
      <div 
        className={cn(
          'flex items-center justify-center bg-muted rounded',
          className
        )}
        style={{ width, height }}
      >
        <span className="text-muted-foreground text-sm">Failed to load image</span>
      </div>
    );
  }

  return (
    <div className={cn('relative overflow-hidden', className)}>
      {/* Placeholder */}
      {!isLoaded && placeholderSrc && (
        <img
          src={placeholderSrc}
          alt=""
          className="absolute inset-0 w-full h-full object-cover filter blur-sm"
          style={{ width, height }}
        />
      )}

      {/* Main image */}
      <img
        ref={imgRef}
        src={loading === 'lazy' ? undefined : optimizedUrls.src}
        srcSet={loading === 'lazy' ? undefined : optimizedUrls.srcSet}
        data-src={loading === 'lazy' ? optimizedUrls.src : undefined}
        data-srcset={loading === 'lazy' ? optimizedUrls.srcSet : undefined}
        alt={alt}
        width={width}
        height={height}
        sizes={sizes || optimizedUrls.sizes}
        onLoad={handleLoad}
        onError={handleError}
        className={cn(
          'transition-opacity duration-300',
          loading === 'lazy' && 'lazy',
          isLoaded ? 'opacity-100' : 'opacity-0',
          className
        )}
        loading={loading}
        decoding="async"
      />
    </div>
  );
};

// Component for hero images with advanced optimization
interface HeroImageProps extends OptimizedImageProps {
  overlay?: boolean;
  overlayColor?: string;
  overlayOpacity?: number;
}

export const HeroImage: React.FC<HeroImageProps> = ({
  overlay = false,
  overlayColor = 'black',
  overlayOpacity = 0.3,
  className,
  ...props
}) => {
  return (
    <div className={cn('relative', className)}>
      <OptimizedImage
        {...props}
        priority={true}
        loading="eager"
        quality={90}
        className="w-full h-full object-cover"
      />
      
      {overlay && (
        <div 
          className="absolute inset-0"
          style={{
            backgroundColor: overlayColor,
            opacity: overlayOpacity,
          }}
        />
      )}
    </div>
  );
};

// Gallery component with lazy loading
interface ImageGalleryProps {
  images: Array<{
    src: string;
    alt: string;
    caption?: string;
  }>;
  columns?: number;
  spacing?: number;
  className?: string;
}

export const ImageGallery: React.FC<ImageGalleryProps> = ({
  images,
  columns = 3,
  spacing = 4,
  className,
}) => {
  return (
    <div 
      className={cn('grid gap-4', className)}
      style={{
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap: `${spacing * 0.25}rem`,
      }}
    >
      {images.map((image, index) => (
        <div key={index} className="group cursor-pointer">
          <OptimizedImage
            src={image.src}
            alt={image.alt}
            loading={index < 6 ? 'eager' : 'lazy'} // Load first 6 eagerly
            className="w-full h-auto rounded-lg transition-transform duration-200 group-hover:scale-105"
            quality={80}
          />
          {image.caption && (
            <p className="mt-2 text-sm text-muted-foreground">{image.caption}</p>
          )}
        </div>
      ))}
    </div>
  );
};

// Avatar component with optimization
interface AvatarImageProps {
  src: string;
  alt: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  fallback?: string;
  className?: string;
}

export const AvatarImage: React.FC<AvatarImageProps> = ({
  src,
  alt,
  size = 'md',
  fallback,
  className,
}) => {
  const sizeMap = {
    sm: 32,
    md: 48,
    lg: 64,
    xl: 96,
  };

  const dimension = sizeMap[size];

  return (
    <OptimizedImage
      src={src}
      alt={alt}
      width={dimension}
      height={dimension}
      quality={90}
      className={cn('rounded-full object-cover', className)}
      placeholder={fallback ? 'empty' : 'blur'}
      onError={() => {
        if (fallback) {
          // Set fallback image
        }
      }}
    />
  );
};

export default OptimizedImage;