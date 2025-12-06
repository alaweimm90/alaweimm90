/**
 * User Behavior Tracking Hook
 * Tracks user interactions: clicks, mouse movements, focus changes, etc.
 * Includes time on page, bounce rate indicators, and engagement metrics
 */

import { useEffect, useRef } from 'react';
import { getAnalytics } from '@/lib/analytics';
import { getErrorTracker } from '@/lib/errorTracking';
import { getConsentManager } from '@/lib/consentManager';

export interface UserBehaviorOptions {
  trackClicks?: boolean;
  trackFocus?: boolean;
  trackMouseMovement?: boolean;
  trackKeyboard?: boolean;
  trackVisibility?: boolean;
  trackExitIntent?: boolean;
  throttleMs?: number;
}

export interface EngagementMetrics {
  totalClicks: number;
  focusChanges: number;
  mouseMoves: number;
  keyboardEvents: number;
  isEngaged: boolean;
  timeOnPage: number;
  lastActivityTime: number;
}

const defaultOptions: UserBehaviorOptions = {
  trackClicks: true,
  trackFocus: true,
  trackMouseMovement: false, // Disabled by default for privacy
  trackKeyboard: false, // Disabled by default for privacy
  trackVisibility: true,
  trackExitIntent: true,
  throttleMs: 200,
};

export const useUserBehavior = (options: UserBehaviorOptions = {}) => {
  const config = { ...defaultOptions, ...options };
  const analytics = getAnalytics();
  const errorTracker = getErrorTracker();
  const consentManager = getConsentManager();
  const {
    trackClicks,
    trackFocus,
    trackMouseMovement,
    trackKeyboard,
    trackVisibility,
    trackExitIntent,
    throttleMs,
  } = config;

  const metricsRef = useRef<EngagementMetrics>({
    totalClicks: 0,
    focusChanges: 0,
    mouseMoves: 0,
    keyboardEvents: 0,
    isEngaged: false,
    timeOnPage: 0,
    lastActivityTime: 0,
  });
  const pageStartTimeRef = useRef<number>(0);
  const lastMouseMoveTimeRef = useRef<number>(0);
  const visibleRef = useRef<boolean>(!document.hidden);

  useEffect(() => {
    metricsRef.current.lastActivityTime = Date.now();
    pageStartTimeRef.current = Date.now();
  }, []);

  useEffect(() => {
    if (!consentManager.hasConsent('analytics')) {
      return;
    }

    // ====================================================================
    // Click Tracking
    // ====================================================================

    const handleClick = (event: MouseEvent) => {
      metricsRef.current.totalClicks++;
      metricsRef.current.isEngaged = true;
      metricsRef.current.lastActivityTime = Date.now();

      const target = event.target as HTMLElement;
      const elementType = target.tagName.toLowerCase();
      const elementClass = target.className || 'no-class';
      const elementId = target.id || 'no-id';

      errorTracker.addUserActionBreadcrumb(
        `Clicked ${elementType}`,
        elementId
      );

      // Track button clicks specifically
      if (elementType === 'button' || elementType === 'a') {
        analytics.buttonClick(
          target.textContent?.substring(0, 50) || elementType,
          {
            element_type: elementType,
            element_class: elementClass,
            element_id: elementId,
          }
        );
      }

      if (import.meta.env.DEV) {
        console.log('User clicked:', {
          type: elementType,
          text: target.textContent?.substring(0, 50),
          id: elementId,
        });
      }
    };

    // ====================================================================
    // Focus/Blur Tracking
    // ====================================================================

    const handleFocus = () => {
      metricsRef.current.focusChanges++;
      metricsRef.current.isEngaged = true;
      metricsRef.current.lastActivityTime = Date.now();

      errorTracker.addBreadcrumb('User focused on page', 'engagement');
    };

    const handleBlur = () => {
      metricsRef.current.lastActivityTime = Date.now();
      errorTracker.addBreadcrumb('User left page focus', 'engagement');
    };

    // ====================================================================
    // Mouse Movement Tracking (Privacy-Conscious)
    // ====================================================================

    const handleMouseMove = () => {
      const now = Date.now();

      if (config.trackMouseMovement && now - lastMouseMoveTimeRef.current > 1000) {
        metricsRef.current.mouseMoves++;
        metricsRef.current.isEngaged = true;
        metricsRef.current.lastActivityTime = now;
        lastMouseMoveTimeRef.current = now;
      }
    };

    // ====================================================================
    // Keyboard Tracking (Privacy-Conscious, no key logging)
    // ====================================================================

    const handleKeydown = () => {
      metricsRef.current.keyboardEvents++;
      metricsRef.current.isEngaged = true;
      metricsRef.current.lastActivityTime = Date.now();
    };

    // ====================================================================
    // Visibility Change Tracking
    // ====================================================================

    const handleVisibilityChange = () => {
      visibleRef.current = !document.hidden;

      if (document.hidden) {
        errorTracker.addBreadcrumb('Page hidden/minimized', 'engagement');
      } else {
        errorTracker.addBreadcrumb('Page visible again', 'engagement');
      }
    };

    // ====================================================================
    // Exit Intent Tracking
    // ====================================================================

    const handleMouseLeave = (event: MouseEvent) => {
      if (event.clientY <= 0) {
        analytics.track('exit_intent', {
          page_path: window.location.pathname,
          time_on_page: Math.round((Date.now() - pageStartTimeRef.current) / 1000),
          event_category: 'engagement',
        });

        errorTracker.addBreadcrumb('User moving mouse outside window', 'engagement');

        if (import.meta.env.DEV) {
          console.log('Exit intent detected');
        }
      }
    };

    // ====================================================================
    // Event Listeners
    // ====================================================================

    if (trackClicks) {
      document.addEventListener('click', handleClick, true);
    }

    if (trackFocus) {
      window.addEventListener('focus', handleFocus);
      window.addEventListener('blur', handleBlur);
    }

    if (trackMouseMovement) {
      document.addEventListener('mousemove', handleMouseMove, { passive: true });
    }

    if (trackKeyboard) {
      document.addEventListener('keydown', handleKeydown, { passive: true });
    }

    if (trackVisibility) {
      document.addEventListener('visibilitychange', handleVisibilityChange);
    }

    if (trackExitIntent) {
      document.addEventListener('mouseleave', handleMouseLeave);
    }

    // ====================================================================
    // Page Unload Handler
    // ====================================================================

    const handleBeforeUnload = () => {
      const timeOnPage = Math.round((Date.now() - pageStartTimeRef.current) / 1000);
      const bounceRate = metricsRef.current.totalClicks === 0 ? 1 : 0;

      analytics.track('page_engagement', {
        time_on_page: timeOnPage,
        total_clicks: metricsRef.current.totalClicks,
        focus_changes: metricsRef.current.focusChanges,
        is_bounce: bounceRate === 1,
        page_path: window.location.pathname,
        event_category: 'engagement',
      });

      if (timeOnPage > 0) {
        analytics.trackTimeOnPage(window.location.pathname, timeOnPage);
      }

      if (bounceRate === 1) {
        analytics.track('bounce', {
          page_path: window.location.pathname,
          event_category: 'engagement',
        });
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    // ====================================================================
    // Periodic Engagement Check
    // ====================================================================

    const engagementCheckInterval = setInterval(() => {
      const timeSinceLastActivity = Date.now() - metricsRef.current.lastActivityTime;

      // If no activity for 30 seconds, consider user disengaged
      if (timeSinceLastActivity > 30000 && metricsRef.current.isEngaged && visibleRef.current) {
        metricsRef.current.isEngaged = false;

        analytics.track('engagement_ended', {
          total_interactions: metricsRef.current.totalClicks +
            metricsRef.current.focusChanges +
            metricsRef.current.keyboardEvents,
          page_path: window.location.pathname,
          event_category: 'engagement',
        });

        if (import.meta.env.DEV) {
          console.log('User engagement ended');
        }
      } else if (timeSinceLastActivity < 5000 && !metricsRef.current.isEngaged && visibleRef.current) {
        metricsRef.current.isEngaged = true;

        analytics.track('engagement_started', {
          page_path: window.location.pathname,
          event_category: 'engagement',
        });

        if (import.meta.env.DEV) {
          console.log('User engagement started');
        }
      }
    }, 5000);

    return () => {
      if (trackClicks) {
        document.removeEventListener('click', handleClick, true);
      }
      if (trackFocus) {
        window.removeEventListener('focus', handleFocus);
        window.removeEventListener('blur', handleBlur);
      }
      if (trackMouseMovement) {
        document.removeEventListener('mousemove', handleMouseMove);
      }
      if (trackKeyboard) {
        document.removeEventListener('keydown', handleKeydown);
      }
      if (trackVisibility) {
        document.removeEventListener('visibilitychange', handleVisibilityChange);
      }
      if (trackExitIntent) {
        document.removeEventListener('mouseleave', handleMouseLeave);
      }

      window.removeEventListener('beforeunload', handleBeforeUnload);
      clearInterval(engagementCheckInterval);
    };
  }, [
    analytics,
    errorTracker,
    consentManager,
    trackClicks,
    trackFocus,
    trackMouseMovement,
    trackKeyboard,
    trackVisibility,
    trackExitIntent,
    throttleMs,
  ]);

  // ========================================================================
  // Getters for engagement metrics
  // ========================================================================

  const getMetrics = (): EngagementMetrics => {
    return {
      ...metricsRef.current,
      timeOnPage: Math.round((Date.now() - pageStartTimeRef.current) / 1000),
    };
  };

  const isUserEngaged = (): boolean => {
    return metricsRef.current.isEngaged && visibleRef.current;
  };

  const getTimeOnPage = (): number => {
    return Math.round((Date.now() - pageStartTimeRef.current) / 1000);
  };

  const getBounceIndicator = (): boolean => {
    return metricsRef.current.totalClicks === 0;
  };

  return {
    getMetrics,
    isUserEngaged,
    getTimeOnPage,
    getBounceIndicator,
  };
};

export default useUserBehavior;
