import { useEffect, useState } from 'react';

/**
 * ScrollIndicator component displays a progress bar showing page scroll position
 *
 * Renders a fixed progress bar at the top of the viewport that fills from left to right
 * as the user scrolls down the page. Updates smoothly with scroll events.
 *
 * @component
 *
 * @example
 * <ScrollIndicator />
 *
 * @remarks
 * - Fixed position at top of viewport
 * - Gradient color from accent to accent/70
 * - Smooth transition on scroll
 * - Z-index 50 to stay above content
 * - Calculates progress: (scrollTop) / (totalHeight - viewportHeight)
 */
const ScrollIndicator = () => {
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    const updateScrollProgress = () => {
      const scrollPx = document.documentElement.scrollTop;
      const winHeightPx =
        document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = scrollPx / winHeightPx;
      setScrollProgress(scrolled);
    };

    window.addEventListener('scroll', updateScrollProgress);
    return () => window.removeEventListener('scroll', updateScrollProgress);
  }, []);

  return (
    <div className="fixed top-0 left-0 w-full h-1 bg-border/30 z-50">
      <div
        className="h-full bg-gradient-to-r from-accent to-accent/70 transition-all duration-150 ease-out"
        style={{ width: `${scrollProgress * 100}%` }}
      />
    </div>
  );
};

export default ScrollIndicator;
