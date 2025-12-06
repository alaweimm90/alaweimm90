/**
 * Analytics utility for tracking user events and metrics
 * Supports multiple analytics providers and custom event tracking
 */

interface AnalyticsEvent {
  name: string;
  properties?: Record<string, any>;
  timestamp?: string;
}

interface AnalyticsProvider {
  track: (event: AnalyticsEvent) => void;
  identify: (userId: string, traits?: Record<string, any>) => void;
  page: (name?: string, properties?: Record<string, any>) => void;
}

class AnalyticsService {
  private providers: AnalyticsProvider[] = [];
  private isEnabled: boolean = true;
  private queue: AnalyticsEvent[] = [];

  constructor() {
    // Check if analytics is enabled (respect DNT, user preferences, etc.)
    this.isEnabled = this.checkAnalyticsEnabled();
  }

  private checkAnalyticsEnabled(): boolean {
    // Respect Do Not Track
    if (navigator.doNotTrack === '1') {
      return false;
    }

    // Check environment
    if (import.meta.env.MODE === 'development') {
      // Log events in development but don't send
      return false;
    }

    return true;
  }

  /**
   * Initialize analytics (idempotent - safe to call multiple times)
   */
  init() {
    // Re-check if analytics should be enabled
    this.isEnabled = this.checkAnalyticsEnabled();
    console.log('[Analytics] Initialized:', this.isEnabled ? 'enabled' : 'disabled');
  }

  /**
   * Add an analytics provider
   */
  addProvider(provider: AnalyticsProvider) {
    this.providers.push(provider);
    // Flush queued events to new provider
    this.flushQueue();
  }

  /**
   * Track a custom event
   */
  trackCustom(eventName: string, properties?: Record<string, any>) {
    const event: AnalyticsEvent = {
      name: eventName,
      properties,
      timestamp: new Date().toISOString()
    };

    if (!this.isEnabled) {
      console.log('[Analytics] Event (not sent):', event);
      return;
    }

    if (this.providers.length === 0) {
      this.queue.push(event);
      return;
    }

    this.providers.forEach(provider => {
      try {
        provider.track(event);
      } catch (error) {
        console.error('[Analytics] Error tracking event:', error);
      }
    });
  }

  /**
   * Track a page view
   */
  trackPage(pageName?: string, properties?: Record<string, any>) {
    if (!this.isEnabled) {
      console.log('[Analytics] Page view (not sent):', pageName, properties);
      return;
    }

    this.providers.forEach(provider => {
      try {
        provider.page(pageName, properties);
      } catch (error) {
        console.error('[Analytics] Error tracking page:', error);
      }
    });
  }

  /**
   * Identify a user
   */
  identify(userId: string, traits?: Record<string, any>) {
    if (!this.isEnabled) {
      console.log('[Analytics] Identify (not sent):', userId, traits);
      return;
    }

    this.providers.forEach(provider => {
      try {
        provider.identify(userId, traits);
      } catch (error) {
        console.error('[Analytics] Error identifying user:', error);
      }
    });
  }

  /**
   * Track page view with path and title
   */
  trackPageView(path: string, title: string) {
    this.trackPage(path, { title });
  }

  /**
   * Track time spent on page
   */
  trackTimeOnPage(seconds: number, path: string) {
    this.trackCustom('time_on_page', { seconds, path });
  }

  /**
   * Track scroll depth percentage
   */
  trackScrollDepth(percent: number) {
    this.trackCustom('scroll_depth', { percent });
  }

  /**
   * Track error occurrences
   */
  trackError(message: string, source?: string) {
    this.trackCustom('error', { message, source });
  }

  /**
   * Track consultation booking
   */
  trackConsultationBooking(tier?: string) {
    this.trackCustom('consultation_booked', { tier });
  }

  /**
   * Track tier reservation
   */
  trackTierReservation(tier: string, price?: number) {
    this.trackCustom('tier_reserved', { tier, price });
  }

  /**
   * Track email signup
   */
  trackEmailSignup(source?: string) {
    this.trackCustom('email_signup', { source });
  }

  /**
   * Track phone call
   */
  trackPhoneCall(source?: string) {
    this.trackCustom('phone_call', { source });
  }

  /**
   * Track hero section engagement
   */
  trackHeroEngagement(action: string, element?: string) {
    this.trackCustom('hero_engagement', { action, element });
  }

  /**
   * Track pricing tier view
   */
  trackPricingView(tier?: string) {
    this.trackCustom('pricing_viewed', { tier });
  }

  /**
   * Track reservation form start
   */
  trackReservationFormStart(tier?: string) {
    this.trackCustom('reservation_form_started', { tier });
  }

  /**
   * Track reservation form completion
   */
  trackReservationFormComplete(tier?: string) {
    this.trackCustom('reservation_form_completed', { tier });
  }

  /**
   * Track payment method submission
   */
  trackPaymentMethodSubmission(tier?: string, method?: string) {
    this.trackCustom('payment_method_submitted', { tier, method });
  }

  /**
   * Track button clicks
   */
  trackButtonClick(buttonText: string, location?: string) {
    this.trackCustom('button_clicked', { buttonText, location });
  }

  /**
   * Track link clicks
   */
  trackLinkClick(linkText: string, destination?: string) {
    this.trackCustom('link_clicked', { linkText, destination });
  }

  /**
   * Track A/B test view
   */
  trackAbTestView(testName: string, variant: string) {
    this.trackCustom('ab_test_viewed', { testName, variant });
  }

  /**
   * Track A/B test conversion
   */
  trackAbTestConversion(testName: string, variant: string, goal?: string) {
    this.trackCustom('ab_test_converted', { testName, variant, goal });
  }

  /**
   * Track custom event with category and action
   */
  trackCustomEvent(category: string, action: string, data?: Record<string, any>) {
    this.trackCustom('custom_event', { category, action, ...data });
  }

  /**
   * Flush queued events
   */
  private flushQueue() {
    while (this.queue.length > 0) {
      const event = this.queue.shift();
      if (event) {
        this.trackCustom(event.name, event.properties);
      }
    }
  }

  /**
   * Enable or disable analytics
   */
  setEnabled(enabled: boolean) {
    this.isEnabled = enabled;
  }
}

// Create singleton instance
export const Analytics = new AnalyticsService();

// Console provider for development
export const ConsoleProvider: AnalyticsProvider = {
  track: (event) => {
    console.log('[Analytics] Event:', event);
  },
  identify: (userId, traits) => {
    console.log('[Analytics] Identify:', userId, traits);
  },
  page: (name, properties) => {
    console.log('[Analytics] Page:', name, properties);
  }
};

// Add console provider in development
if (import.meta.env.MODE === 'development') {
  Analytics.addProvider(ConsoleProvider);
}

// Export common tracking helpers
export const trackEvent = (name: string, properties?: Record<string, any>) => {
  Analytics.trackCustom(name, properties);
};

export const trackPageView = (pageName?: string, properties?: Record<string, any>) => {
  Analytics.trackPage(pageName, properties);
};

export const identifyUser = (userId: string, traits?: Record<string, any>) => {
  Analytics.identify(userId, traits);
};

export default Analytics;
