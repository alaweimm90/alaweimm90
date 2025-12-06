import React, { useState, useRef, useEffect, useCallback } from 'react';
import { cn } from '@/lib/utils';

interface OptimizedImageAdvancedProps extends Omit<React.ImgHTMLAttributes<HTMLImageElement>, 'src'> {
  src: string;
  alt: string;
  width?: number;
  height?: number;
  quality?: number;
  format?: 'webp' | 'avif' | 'auto';
  placeholder?: 'blur' | 'empty' | string;
  priority?: boolean;
  sizes?: string;
  className?: string;
  onLoad?: () => void;
  onError?: () => void;
}

// WebP and AVIF support detection
const hasWebPSupport = (() => {
  if (typeof window === 'undefined') return false;
  
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  
  return canvas.toDataURL('image/webp').indexOf('image/webp') === 5;
})();

const hasAVIFSupport = (() => {
  if (typeof window === 'undefined') return false;
  
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  
  try {
    return canvas.toDataURL('image/avif').indexOf('image/avif') === 5;
  } catch {
    return false;
  }
})();

// Image optimization service (would be replaced with actual service like Cloudinary, ImageKit, etc.)
const optimizeImageUrl = (
  src: string, 
  width?: number, 
  height?: number, 
  quality = 80, 
  format?: string
): string => {
  // In production, this would integrate with your image optimization service
  // For now, we'll just add query parameters that could be processed by a CDN
  
  const url = new URL(src, window.location.origin);
  
  if (width) url.searchParams.set('w', width.toString());
  if (height) url.searchParams.set('h', height.toString());
  if (quality !== 80) url.searchParams.set('q', quality.toString());
  if (format) url.searchParams.set('f', format);
  
  return url.toString();
};

// Generate responsive image URLs
const generateSrcSet = (
  src: string,
  width?: number,
  quality?: number,
  format?: string
): string => {
  if (!width) return '';
  
  const sizes = [0.5, 1, 1.5, 2]; // Different pixel densities
  
  return sizes
    .map(multiplier => {
      const scaledWidth = Math.round(width * multiplier);
      const optimizedUrl = optimizeImageUrl(src, scaledWidth, undefined, quality, format);
      return `${optimizedUrl} ${multiplier}x`;
    })
    .join(', ');
};

export const OptimizedImageAdvanced: React.FC<OptimizedImageAdvancedProps> = ({
  src,
  alt,
  width,
  height,
  quality = 80,
  format = 'auto',
  placeholder = 'blur',
  priority = false,
  sizes,
  className,
  onLoad,
  onError,
  ...props
}) => {
  const [isLoading, setIsLoading] = useState(true);
  const [isInView, setIsInView] = useState(priority);
  const [hasError, setHasError] = useState(false);
  const [currentSrc, setCurrentSrc] = useState<string>('');
  
  const imgRef = useRef<HTMLImageElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Determine optimal format
  const getOptimalFormat = useCallback(() => {
    if (format !== 'auto') return format;
    
    if (hasAVIFSupport) return 'avif';
    if (hasWebPSupport) return 'webp';
    return undefined; // Use original format
  }, [format]);

  // Generate optimized source URLs
  const generateSources = useCallback(() => {
    const optimalFormat = getOptimalFormat();
    const baseUrl = optimizeImageUrl(src, width, height, quality, optimalFormat);
    const srcSet = generateSrcSet(src, width, quality, optimalFormat);
    
    return {
      src: baseUrl,
      srcSet: srcSet || undefined
    };
  }, [src, width, height, quality, getOptimalFormat]);

  // Intersection Observer for lazy loading
  useEffect(() => {
    if (priority || !imgRef.current) return;

    observerRef.current = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observerRef.current?.disconnect();
        }
      },
      {
        threshold: 0.1,
        rootMargin: '50px'
      }
    );

    observerRef.current.observe(imgRef.current);

    return () => {
      observerRef.current?.disconnect();
    };
  }, [priority]);

  // Load image when in view
  useEffect(() => {
    if (!isInView) return;

    const { src: optimizedSrc } = generateSources();
    setCurrentSrc(optimizedSrc);
  }, [isInView, generateSources]);

  // Handle image load
  const handleLoad = useCallback(() => {
    setIsLoading(false);
    onLoad?.();
  }, [onLoad]);

  // Handle image error
  const handleError = useCallback(() => {
    setHasError(true);
    setIsLoading(false);
    
    // Try fallback to original image
    if (currentSrc !== src) {
      setCurrentSrc(src);
      setHasError(false);
    } else {
      onError?.();
    }
  }, [currentSrc, src, onError]);

  // Generate blur placeholder
  const generateBlurPlaceholder = useCallback(() => {
    if (placeholder !== 'blur') return undefined;
    
    // Generate a tiny blurred version (would be replaced with actual blur hash in production)
    const canvas = document.createElement('canvas');
    canvas.width = 40;
    canvas.height = 30;
    
    const ctx = canvas.getContext('2d');
    if (ctx) {
      // Create a simple gradient as placeholder
      const gradient = ctx.createLinearGradient(0, 0, 40, 30);
      gradient.addColorStop(0, '#f3f4f6');
      gradient.addColorStop(1, '#e5e7eb');
      
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 40, 30);
    }
    
    return canvas.toDataURL();
  }, [placeholder]);

  const blurDataUrl = generateBlurPlaceholder();
  const { srcSet } = generateSources();

  return (
    <div 
      className={cn('relative overflow-hidden', className)}
      style={{ 
        width: width ? `${width}px` : undefined,
        height: height ? `${height}px` : undefined
      }}
    >
      {/* Blur placeholder */}
      {isLoading && blurDataUrl && (
        <img
          src={blurDataUrl}
          alt=""
          className="absolute inset-0 w-full h-full object-cover filter blur-sm scale-110"
          aria-hidden="true"
        />
      )}
      
      {/* Loading placeholder */}
      {isLoading && placeholder === 'empty' && (
        <div className="absolute inset-0 bg-gradient-to-br from-gray-100 to-gray-200 animate-pulse" />
      )}
      
      {/* Custom placeholder */}
      {isLoading && typeof placeholder === 'string' && placeholder !== 'blur' && placeholder !== 'empty' && (
        <img
          src={placeholder}
          alt=""
          className="absolute inset-0 w-full h-full object-cover opacity-50"
          aria-hidden="true"
        />
      )}
      
      {/* Loading spinner */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
        </div>
      )}
      
      {/* Main image */}
      {(isInView || priority) && (
        <picture>
          {/* AVIF support */}
          {hasAVIFSupport && format === 'auto' && (
            <source
              srcSet={generateSrcSet(src, width, quality, 'avif')}
              sizes={sizes}
              type="image/avif"
            />
          )}
          
          {/* WebP support */}
          {hasWebPSupport && (format === 'auto' || format === 'webp') && (
            <source
              srcSet={generateSrcSet(src, width, quality, 'webp')}
              sizes={sizes}
              type="image/webp"
            />
          )}
          
          {/* Fallback */}
          <img
            ref={imgRef}
            src={currentSrc}
            srcSet={srcSet}
            sizes={sizes}
            alt={alt}
            className={cn(
              'transition-opacity duration-500',
              isLoading ? 'opacity-0' : 'opacity-100',
              'w-full h-full object-cover'
            )}
            loading={priority ? 'eager' : 'lazy'}
            decoding="async"
            onLoad={handleLoad}
            onError={handleError}
            {...props}
          />
        </picture>
      )}
      
      {/* Error state */}
      {hasError && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
          <div className="text-center text-gray-500">
            <div className="text-2xl mb-2">📷</div>
            <div className="text-sm">Image unavailable</div>
          </div>
        </div>
      )}
      
      {/* Progressive enhancement indicators */}
      {process.env.NODE_ENV === 'development' && (
        <div className="absolute top-2 left-2 text-xs bg-black bg-opacity-50 text-white px-2 py-1 rounded">
          {hasAVIFSupport ? 'AVIF' : hasWebPSupport ? 'WebP' : 'Original'}
          {isLoading && ' • Loading'}
        </div>
      )}
    </div>
  );
};

// Hook for managing image preloading
export const useImagePreload = (sources: string[]) => {
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());
  
  useEffect(() => {
    const preloadImages = sources.map(src => {
      const img = new Image();
      
      return new Promise<string>((resolve, reject) => {
        img.onload = () => resolve(src);
        img.onerror = () => reject(src);
        img.src = src;
      });
    });
    
    Promise.allSettled(preloadImages).then(results => {
      const loaded = new Set<string>();
      
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          loaded.add(sources[index]);
        }
      });
      
      setLoadedImages(loaded);
    });
  }, [sources]);
  
  return loadedImages;
};

// Component for image galleries with optimized loading
export const ImageGallery: React.FC<{
  images: Array<{
    src: string;
    alt: string;
    width?: number;
    height?: number;
  }>;
  className?: string;
}> = ({ images, className }) => {
  const visibleImages = images.slice(0, 6); // Load first 6 images
  const preloadSources = visibleImages.map(img => img.src);
  const loadedImages = useImagePreload(preloadSources);
  
  return (
    <div className={cn('grid grid-cols-2 md:grid-cols-3 gap-4', className)}>
      {images.map((image, index) => (
        <OptimizedImageAdvanced
          key={`${image.src}-${index}`}
          src={image.src}
          alt={image.alt}
          width={image.width}
          height={image.height}
          priority={index < 3} // Prioritize first 3 images
          className="aspect-square rounded-lg"
          placeholder={loadedImages.has(image.src) ? 'empty' : 'blur'}
        />
      ))}
    </div>
  );
};