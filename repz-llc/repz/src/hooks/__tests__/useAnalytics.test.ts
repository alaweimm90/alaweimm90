import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook } from '@testing-library/react'
import React from 'react'
import { BrowserRouter } from 'react-router-dom'

// Mock the Analytics modules with inline factory functions
vi.mock('@/lib/analytics', () => ({
  Analytics: {
    init: vi.fn(),
    addProvider: vi.fn(),
    setEnabled: vi.fn(),
    trackPageView: vi.fn(),
    trackTimeOnPage: vi.fn(),
    trackScrollDepth: vi.fn(),
    trackPage: vi.fn(),
    trackError: vi.fn(),
    trackConsultationBooking: vi.fn(),
    trackTierReservation: vi.fn(),
    trackEmailSignup: vi.fn(),
    trackPhoneCall: vi.fn(),
    trackHeroEngagement: vi.fn(),
    trackPricingView: vi.fn(),
    trackReservationFormStart: vi.fn(),
    trackReservationFormComplete: vi.fn(),
    trackPaymentMethodSubmission: vi.fn(),
    trackButtonClick: vi.fn(),
    trackLinkClick: vi.fn(),
    trackAbTestView: vi.fn(),
    trackAbTestConversion: vi.fn(),
    trackCustom: vi.fn(),
    trackCustomEvent: vi.fn(),
    identify: vi.fn(),
  },
}))

// Import after mock is set up
import { useAnalytics } from '../useAnalytics'
import { Analytics } from '@/lib/analytics'

beforeEach(() => {
  vi.clearAllMocks()
})

describe('useAnalytics', () => {
  it('tracks tier reservation correctly', () => {
    const wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(BrowserRouter, null, children)
    const { result } = renderHook(() => useAnalytics(), { wrapper })

    result.current.trackConversion.tierReservation('core-program', 96)

    expect(Analytics.trackTierReservation).toHaveBeenCalledWith('core-program', 96)
  })

  it('tracks custom events correctly', () => {
    const wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(BrowserRouter, null, children)
    const { result } = renderHook(() => useAnalytics(), { wrapper })

    result.current.trackCustom('user_interaction', 'button_click', { tier: 'core-program' })

    expect(Analytics.trackCustomEvent).toHaveBeenCalledWith(
      'user_interaction',
      'button_click',
      { tier: 'core-program' }
    )
  })

  it('tracks funnel progression', () => {
    const wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(BrowserRouter, null, children)
    const { result } = renderHook(() => useAnalytics(), { wrapper })

    result.current.trackFunnel.pricingView('core-program')

    expect(Analytics.trackPricingView).toHaveBeenCalledWith('core-program')
  })
})
