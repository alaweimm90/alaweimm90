import React, { useEffect, useRef } from 'react';

interface LiveRegionProps {
  message: string;
  priority?: 'polite' | 'assertive';
  clearAfter?: number;
}

export function LiveRegion({ message, priority = 'polite', clearAfter = 3000 }: LiveRegionProps) {
  const regionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (message && regionRef.current) {
      regionRef.current.textContent = message;
      
      if (clearAfter > 0) {
        const timer = setTimeout(() => {
          if (regionRef.current) {
            regionRef.current.textContent = '';
          }
        }, clearAfter);
        
        return () => clearTimeout(timer);
      }
    }
  }, [message, clearAfter]);

  return (
    <div
      ref={regionRef}
      aria-live={priority}
      aria-atomic="true"
      className="sr-only"
    />
  );
}

// Global live region hook for announcements
export function useAnnouncer() {
  const announce = (message: string, priority: 'polite' | 'assertive' = 'polite') => {
    const event = new CustomEvent('a11y-announce', {
      detail: { message, priority }
    });
    window.dispatchEvent(event);
  };

  return { announce };
}