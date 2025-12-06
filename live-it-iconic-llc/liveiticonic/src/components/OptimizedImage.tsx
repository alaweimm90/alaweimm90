/**
 * OptimizedImage Component
 * Provides responsive, lazy-loaded images with WebP/AVIF support and blur placeholders
 * Helps achieve LCP and CLS targets through:
 * - Modern image formats (AVIF/WebP) with JPEG fallback
 * - Responsive sizing with srcset
 * - Lazy loading with intersection observer
 * - Blur-up placeholder effect
 * - Explicit dimensions to prevent CLS
 * - Priority loading for critical images (LCP)
 */

import React, { useEffect, useRef, useState } from 'react';

interface OptimizedImageProps {
  /**
   * Source of the image (imported)
   * Used as fallback for browsers without modern format support
   */
  src: string;
  /**
   * WebP format source for better compression
   */
  srcWebp?: string;
  /**
   * AVIF format source for best compression
   */
  srcAvif?: string;
  /**
   * Alt text for accessibility
   */
  alt: string;
  /**
   * CSS classes to apply
   */
  className?: string;
  /**
   * Width in pixels for explicit sizing (prevents CLS)
   */
  width?: number;
  /**
   * Height in pixels for explicit sizing (prevents CLS)
   */
  height?: number;
  /**
   * Responsive sizes attribute for srcset
   * @example "(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
   */
  sizes?: string;
  /**
   * Load image immediately (priority=true for LCP images)
   */
  priority?: boolean;
  /**
   * Blur-up placeholder image (tiny base64 encoded image or low-res image URL)
   */
  placeholder?: string;
  /**
   * Enable fade-in animation after load
   */
  fadeIn?: boolean;
  /**
   * On load callback
   */
  onLoad?: () => void;
  /**
   * On error callback
   */
  onError?: () => void;
  /**
   * Custom responsive image sizes
   * @example {
   *   mobile: { src: 'mobile.avif', size: 400 },
   *   tablet: { src: 'tablet.avif', size: 800 },
   *   desktop: { src: 'desktop.avif', size: 1200 }
   * }
   */
  responsiveImages?: {
    [key: string]: { src: string; size: number };
  };
}

const OptimizedImage = React.forwardRef<HTMLImageElement, OptimizedImageProps>(
  (
    {
      src,
      srcWebp,
      srcAvif,
      alt,
      className = '',
      width,
      height,
      sizes,
      priority = false,
      placeholder,
      fadeIn = true,
      onLoad,
      onError,
      responsiveImages,
    },
    ref
  ) => {
    const [isLoaded, setIsLoaded] = useState(false);
    const [hasError, setHasError] = useState(false);
    const imgRef = useRef<HTMLImageElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    // Use the ref passed from parent or our internal ref
    const finalRef = (ref as React.MutableRefObject<HTMLImageElement | null>) || imgRef;

    // Intersection Observer for lazy loading
    useEffect(() => {
      if (priority || !containerRef.current) {
        return;
      }

      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              // Trigger image load by setting data attributes to src
              if (imgRef.current) {
                if (imgRef.current.dataset.srcAvif) {
                  imgRef.current.srcset = `${imgRef.current.dataset.srcAvif} 1x`;
                }
                if (imgRef.current.dataset.srcWebp) {
                  // Update srcset for WebP
                  if (imgRef.current.srcset) {
                    imgRef.current.srcset += `, ${imgRef.current.dataset.srcWebp} 1x`;
                  } else {
                    imgRef.current.srcset = `${imgRef.current.dataset.srcWebp} 1x`;
                  }
                }
                if (imgRef.current.dataset.src) {
                  imgRef.current.src = imgRef.current.dataset.src;
                }
              }
              observer.unobserve(entry.target);
            }
          });
        },
        {
          rootMargin: '50px', // Start loading 50px before entering viewport
        }
      );

      observer.observe(containerRef.current);

      return () => {
        if (containerRef.current) {
          observer.unobserve(containerRef.current);
        }
      };
    }, [priority]);

    const handleLoad = () => {
      setIsLoaded(true);
      onLoad?.();
    };

    const handleError = () => {
      setHasError(true);
      onError?.();
    };

    // Calculate aspect ratio for responsive images
    const aspectRatio = width && height ? (height / width) * 100 : undefined;

    // Build picture element for multiple format support
    const pictureContent = (
      <picture>
        {/* AVIF format - best compression (Chrome, Firefox, Safari 16+) */}
        {srcAvif && (
          <source
            srcSet={priority ? srcAvif : (responsiveImages?.desktop?.src || srcAvif)}
            type="image/avif"
          />
        )}
        {/* WebP format - good compression (Chrome, Firefox, Edge) */}
        {srcWebp && (
          <source
            srcSet={priority ? srcWebp : (responsiveImages?.tablet?.src || srcWebp)}
            type="image/webp"
          />
        )}
        {/* Fallback to JPEG */}
        <img
          ref={finalRef}
          src={priority ? src : undefined}
          data-src={!priority ? src : undefined}
          data-srcWebp={!priority ? srcWebp : undefined}
          data-srcAvif={!priority ? srcAvif : undefined}
          alt={alt}
          width={width}
          height={height}
          sizes={sizes}
          className={`${className} ${
            fadeIn && isLoaded ? 'fade-in' : ''
          } transition-opacity duration-300`}
          loading={priority ? 'eager' : 'lazy'}
          onLoad={handleLoad}
          onError={handleError}
          decoding="async"
          {...(responsiveImages && {
            srcSet: priority
              ? Object.values(responsiveImages)
                  .map((img) => `${img.src} ${img.size}w`)
                  .join(', ')
              : undefined,
          })}
        />
      </picture>
    );

    // Wrapper with aspect ratio to prevent CLS
    if (width && height) {
      return (
        <div
          ref={containerRef}
          className="relative overflow-hidden"
          style={{
            paddingBottom: `${aspectRatio}%`,
            backgroundColor: hasError ? '#f3f4f6' : 'transparent',
          }}
        >
          {/* Placeholder blur effect */}
          {placeholder && !isLoaded && (
            <div
              className="absolute inset-0 bg-cover bg-center blur-sm"
              style={{
                backgroundImage: `url(${placeholder})`,
                opacity: isLoaded ? 0 : 1,
                transition: 'opacity 0.3s ease-out',
              }}
              aria-hidden="true"
            />
          )}

          <div className="absolute inset-0">{pictureContent}</div>
        </div>
      );
    }

    return (
      <div
        ref={containerRef}
        className={hasError ? 'bg-gray-200 rounded' : ''}
      >
        {/* Placeholder blur effect for non-fixed-size images */}
        {placeholder && !isLoaded && (
          <div
            className="absolute inset-0 bg-cover bg-center blur-sm"
            style={{
              backgroundImage: `url(${placeholder})`,
              opacity: isLoaded ? 0 : 1,
              transition: 'opacity 0.3s ease-out',
            }}
            aria-hidden="true"
          />
        )}

        {pictureContent}
      </div>
    );
  }
);

OptimizedImage.displayName = 'OptimizedImage';

export default OptimizedImage;
