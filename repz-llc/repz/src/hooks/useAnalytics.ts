import { useEffect, useRef, useCallback } from 'react';
import { useLocation } from 'react-router-dom';
import { Analytics } from '@/lib/analytics';

export const useAnalytics = () => {
  const location = useLocation();
  const timeOnPageStart = useRef<number>(Date.now());
  const scrollDepthTracked = useRef<Set<number>>(new Set());

  // Initialize analytics on mount
  useEffect(() => {
    Analytics.init();
  }, []);

  // Track page views
  useEffect(() => {
    const pagePath = location.pathname + location.search;
    const pageTitle = document.title;
    
    Analytics.trackPageView(pagePath, pageTitle);
    
    // Reset time tracking for new page
    timeOnPageStart.current = Date.now();
    scrollDepthTracked.current.clear();

    // Track time on page when leaving
    return () => {
      const timeOnPage = Math.round((Date.now() - timeOnPageStart.current) / 1000);
      Analytics.trackTimeOnPage(timeOnPage, pagePath);
    };
  }, [location]);

  // Track scroll depth
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrollPercent = Math.round((scrollTop / docHeight) * 100);

      // Track milestone percentages
      const milestones = [25, 50, 75, 90, 100];
      milestones.forEach(milestone => {
        if (scrollPercent >= milestone && !scrollDepthTracked.current.has(milestone)) {
          scrollDepthTracked.current.add(milestone);
          Analytics.trackScrollDepth(milestone);
        }
      });
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [location]);

  // Track errors
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
      Analytics.trackError(event.error?.message || 'Unknown error', event.filename);
    };

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      Analytics.trackError(event.reason?.message || 'Unhandled promise rejection', 'promise');
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);

  // Analytics methods
  const trackConversion = useCallback(() => ({
    consultationBooking: (tier?: string) => Analytics.trackConsultationBooking(tier),
    tierReservation: (tier: string, price: number) => Analytics.trackTierReservation(tier, price),
    emailSignup: (source: string) => Analytics.trackEmailSignup(source),
    phoneCall: (source: string) => Analytics.trackPhoneCall(source),
  }), []);

  const trackFunnel = useCallback(() => ({
    heroEngagement: (action: string, element?: string) => Analytics.trackHeroEngagement(action, element),
    pricingView: (tier?: string) => Analytics.trackPricingView(tier),
    reservationFormStart: (tier: string) => Analytics.trackReservationFormStart(tier),
    reservationFormComplete: (tier: string) => Analytics.trackReservationFormComplete(tier),
    paymentMethodSubmission: (tier: string, method: string) => Analytics.trackPaymentMethodSubmission(tier, method),
  }), []);

  const trackInteraction = useCallback(() => ({
    buttonClick: (buttonText: string, location: string) => Analytics.trackButtonClick(buttonText, location),
    linkClick: (linkText: string, destination: string) => Analytics.trackLinkClick(linkText, destination),
  }), []);

  const trackAbTest = useCallback(() => ({
    view: (testName: string, variant: string) => Analytics.trackAbTestView(testName, variant),
    conversion: (testName: string, variant: string, goal: string) => Analytics.trackAbTestConversion(testName, variant, goal),
  }), []);

  const trackCustom = useCallback((category: string, action: string, data?: Record<string, unknown>) => {
    Analytics.trackCustomEvent(category, action, data);
  }, []);

  return {
    trackConversion: trackConversion(),
    trackFunnel: trackFunnel(),
    trackInteraction: trackInteraction(),
    trackAbTest: trackAbTest(),
    trackCustom,
  };
};