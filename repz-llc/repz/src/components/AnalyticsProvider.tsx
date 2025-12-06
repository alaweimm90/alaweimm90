import React, { createContext, useContext, ReactNode, useMemo } from 'react';
import { Analytics } from '@/lib/analytics';

interface AnalyticsContextType {
  trackConversion: {
    consultationBooking: (tier?: string) => void;
    tierReservation: (tier: string, price: number) => void;
    emailSignup: (source: string) => void;
    phoneCall: (source: string) => void;
  };
  trackFunnel: {
    heroEngagement: (action: string, element?: string) => void;
    pricingView: (tier?: string) => void;
    reservationFormStart: (tier: string) => void;
    reservationFormComplete: (tier: string) => void;
    paymentMethodSubmission: (tier: string, method: string) => void;
  };
  trackInteraction: {
    buttonClick: (buttonText: string, location: string) => void;
    linkClick: (linkText: string, destination: string) => void;
  };
  trackAbTest: {
    view: (testName: string, variant: string) => void;
    conversion: (testName: string, variant: string, goal: string) => void;
  };
  trackCustom: (category: string, action: string, data?: Record<string, unknown>) => void;
}

const AnalyticsContext = createContext<AnalyticsContextType | undefined>(undefined);

export const useAnalyticsContext = () => {
  const context = useContext(AnalyticsContext);
  if (!context) {
    // Safe fallback: no-op analytics to prevent runtime crashes if provider not mounted
    const noop = () => {};
    if (typeof window !== 'undefined') {
      console.warn('[Analytics] Provider missing – using no-op analytics.');
    }
    return {
      trackConversion: {
        consultationBooking: noop,
        tierReservation: noop,
        emailSignup: noop,
        phoneCall: noop,
      },
      trackFunnel: {
        heroEngagement: noop,
        pricingView: noop,
        reservationFormStart: noop,
        reservationFormComplete: noop,
        paymentMethodSubmission: noop,
      },
      trackInteraction: {
        buttonClick: noop,
        linkClick: noop,
      },
      trackAbTest: {
        view: noop,
        conversion: noop,
      },
      trackCustom: noop,
    } as AnalyticsContextType;
  }
  return context;
};

interface AnalyticsProviderProps {
  children: ReactNode;
}

export const AnalyticsProvider: React.FC<AnalyticsProviderProps> = ({ children }) => {
  // Use Analytics singleton directly instead of useAnalytics hook
  // This avoids calling useLocation() at provider level, which causes issues in tests
  const analyticsValue = useMemo(() => ({
    trackConversion: Analytics.trackConversion,
    trackFunnel: Analytics.trackFunnel,
    trackInteraction: Analytics.trackInteraction,
    trackAbTest: Analytics.trackAbTest,
    trackCustom: Analytics.trackCustom,
  }), []);

  return (
    <AnalyticsContext.Provider value={analyticsValue as AnalyticsContextType}>
      {children}
    </AnalyticsContext.Provider>
  );
};