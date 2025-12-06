/**
 * Scroll Depth Tracking Hook
 * Tracks user scroll depth milestones: 25%, 50%, 75%, 100%
 */

import { useEffect, useRef } from 'react';
import { getAnalytics } from '@/lib/analytics';
import { getConsentManager } from '@/lib/consentManager';

export interface ScrollDepthOptions {
  thresholds?: number[]; // Custom thresholds (default: [25, 50, 75, 100])
  trackTimeOnPage?: boolean;
  throttleMs?: number;
}

export const useScrollDepth = (options: ScrollDepthOptions = {}) => {
  const {
    thresholds = [25, 50, 75, 100],
    trackTimeOnPage = true,
    throttleMs = 100,
  } = options;

  const analytics = getAnalytics();
  const consentManager = getConsentManager();

  const trackedDepthsRef = useRef<Set<number>>(new Set());
  const lastScrollTimeRef = useRef<number>(0);
  const pageStartTimeRef = useRef<number>(0);
  const throttleTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    lastScrollTimeRef.current = Date.now();
    pageStartTimeRef.current = Date.now();
    if (!consentManager.hasConsent('analytics')) {
      return;
    }

    const handleScroll = () => {
      // Throttle scroll event handler
      if (throttleTimeoutRef.current) {
        clearTimeout(throttleTimeoutRef.current);
      }

      throttleTimeoutRef.current = setTimeout(() => {
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        const scrollTop = window.scrollY || document.documentElement.scrollTop;

        // Calculate scroll depth percentage
        const maxScroll = documentHeight - windowHeight;
        const scrollDepth = maxScroll > 0 ? Math.round((scrollTop / maxScroll) * 100) : 100;

        // Check thresholds
        thresholds.forEach(threshold => {
          if (scrollDepth >= threshold && !trackedDepthsRef.current.has(threshold)) {
            trackedDepthsRef.current.add(threshold);

            analytics.trackScrollDepth(threshold);

            if (import.meta.env.DEV) {
              console.log(`Scroll depth reached: ${threshold}%`);
            }
          }
        });

        lastScrollTimeRef.current = Date.now();
      }, throttleMs);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });

    // Track page exit (with time on page)
    const handleBeforeUnload = () => {
      if (trackTimeOnPage) {
        const timeOnPage = Math.round((Date.now() - pageStartTimeRef.current) / 1000);

        if (timeOnPage > 0) {
          analytics.trackTimeOnPage(window.location.pathname, timeOnPage);
        }
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('beforeunload', handleBeforeUnload);

      if (throttleTimeoutRef.current) {
        clearTimeout(throttleTimeoutRef.current);
      }
    };
  }, [analytics, consentManager, thresholds, trackTimeOnPage, throttleMs]);

  // Expose getters for current scroll depth
  const getCurrentScrollDepth = (): number => {
    const windowHeight = window.innerHeight;
    const documentHeight = document.documentElement.scrollHeight;
    const scrollTop = window.scrollY || document.documentElement.scrollTop;

    const maxScroll = documentHeight - windowHeight;
    return maxScroll > 0 ? Math.round((scrollTop / maxScroll) * 100) : 100;
  };

  const getTrackedDepths = (): number[] => {
    return Array.from(trackedDepthsRef.current).sort((a, b) => a - b);
  };

  const hasReachedDepth = (threshold: number): boolean => {
    return trackedDepthsRef.current.has(threshold);
  };

  return {
    getCurrentScrollDepth,
    getTrackedDepths,
    hasReachedDepth,
  };
};

export default useScrollDepth;
