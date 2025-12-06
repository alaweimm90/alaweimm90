/**
 * Analytics Mock Utilities
 * Reusable mocks for testing components that use Analytics
 */

import { vi } from 'vitest';

/**
 * Creates a complete Analytics service mock with all tracking methods
 */
export const createAnalyticsMock = () => ({
  // Core methods
  init: vi.fn(),
  addProvider: vi.fn(),
  setEnabled: vi.fn(),

  // Page tracking
  trackPageView: vi.fn(),
  trackTimeOnPage: vi.fn(),
  trackScrollDepth: vi.fn(),
  trackPage: vi.fn(),

  // Error tracking
  trackError: vi.fn(),

  // Conversion tracking
  trackConsultationBooking: vi.fn(),
  trackTierReservation: vi.fn(),
  trackEmailSignup: vi.fn(),
  trackPhoneCall: vi.fn(),

  // Funnel tracking
  trackHeroEngagement: vi.fn(),
  trackPricingView: vi.fn(),
  trackReservationFormStart: vi.fn(),
  trackReservationFormComplete: vi.fn(),
  trackPaymentMethodSubmission: vi.fn(),

  // Interaction tracking
  trackButtonClick: vi.fn(),
  trackLinkClick: vi.fn(),

  // A/B Testing
  trackAbTestView: vi.fn(),
  trackAbTestConversion: vi.fn(),

  // Custom tracking
  trackCustom: vi.fn(),
  trackCustomEvent: vi.fn(),

  // User identification
  identify: vi.fn(),
});

/**
 * Mocks the Analytics module for use in tests
 *
 * @example
 * ```typescript
 * import { mockAnalytics } from '@/test/mocks/analytics';
 *
 * describe('MyComponent', () => {
 *   const analyticsMock = mockAnalytics();
 *
 *   it('tracks page view', () => {
 *     render(<MyComponent />);
 *     expect(analyticsMock.trackPageView).toHaveBeenCalled();
 *   });
 * });
 * ```
 */
export const mockAnalytics = () => {
  const mock = createAnalyticsMock();

  vi.mock('@/lib/analytics', () => ({
    Analytics: mock,
  }));

  vi.mock('@/utils/analytics', () => ({
    Analytics: mock,
    trackEvent: vi.fn((name, props) => mock.trackCustom(name, props)),
    trackPageView: vi.fn((page, props) => mock.trackPage(page, props)),
    identifyUser: vi.fn((id, traits) => mock.identify(id, traits)),
  }));

  return mock;
};

/**
 * Resets all Analytics mocks
 */
export const resetAnalyticsMocks = (mock: ReturnType<typeof createAnalyticsMock>) => {
  Object.values(mock).forEach(fn => {
    if (vi.isMockFunction(fn)) {
      fn.mockClear();
    }
  });
};

/**
 * Asserts that Analytics was initialized
 */
export const expectAnalyticsInitialized = (mock: ReturnType<typeof createAnalyticsMock>) => {
  expect(mock.init).toHaveBeenCalled();
};

/**
 * Asserts that a specific event was tracked
 */
export const expectEventTracked = (
  mock: ReturnType<typeof createAnalyticsMock>,
  eventName: string,
  properties?: Record<string, any>
) => {
  expect(mock.trackCustom).toHaveBeenCalledWith(
    eventName,
    properties ? expect.objectContaining(properties) : expect.anything()
  );
};
