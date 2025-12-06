import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import React from 'react'
import { useAnalyticsContext, AnalyticsProvider } from '@/components/AnalyticsProvider'

// Mock the Analytics lib
vi.mock('@/lib/analytics', () => ({
  Analytics: {
    init: vi.fn(),
    trackPageView: vi.fn(),
    trackTimeOnPage: vi.fn(),
    trackScrollDepth: vi.fn(),
    trackError: vi.fn(),
    trackConversion: {
      consultationBooking: vi.fn(),
      tierReservation: vi.fn(),
      emailSignup: vi.fn(),
      phoneCall: vi.fn(),
    },
    trackFunnel: {
      heroEngagement: vi.fn(),
      pricingView: vi.fn(),
      reservationFormStart: vi.fn(),
      reservationFormComplete: vi.fn(),
      paymentMethodSubmission: vi.fn(),
    },
    trackInteraction: {
      buttonClick: vi.fn(),
      linkClick: vi.fn(),
    },
    trackAbTest: {
      view: vi.fn(),
      conversion: vi.fn(),
    },
    trackCustom: vi.fn(),
  }
}))

// Mock useLocation from react-router-dom
vi.mock('react-router-dom', () => ({
  useLocation: () => ({
    pathname: '/',
    search: '',
    hash: '',
    state: null,
  }),
}))

// Test component that uses analytics context
const TestComponent = () => {
  const analytics = useAnalyticsContext()

  return (
    <div>
      <button onClick={() => analytics.trackConversion.tierReservation('core', 96)}>
        Track Tier
      </button>
      <button onClick={() => analytics.trackCustom('test', 'click', { page: 'home' })}>
        Track Custom
      </button>
      <button onClick={() => analytics.trackFunnel.pricingView('adaptive')}>
        Track Funnel
      </button>
    </div>
  )
}

describe('AnalyticsProvider', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('provides analytics context to children', () => {
    render(<AnalyticsProvider><TestComponent /></AnalyticsProvider>)

    expect(screen.getByText('Track Tier')).toBeInTheDocument()
    expect(screen.getByText('Track Custom')).toBeInTheDocument()
    expect(screen.getByText('Track Funnel')).toBeInTheDocument()
  })

  it('tracks tier reservations correctly', async () => {
    const user = userEvent.setup()
    const { Analytics } = await import('@/lib/analytics')

    render(<AnalyticsProvider><TestComponent /></AnalyticsProvider>)

    await user.click(screen.getByText('Track Tier'))

    expect(Analytics.trackConversion.tierReservation).toHaveBeenCalledWith('core', 96)
  })

  it('tracks custom events with metadata', async () => {
    const user = userEvent.setup()
    const { Analytics } = await import('@/lib/analytics')

    render(<AnalyticsProvider><TestComponent /></AnalyticsProvider>)

    await user.click(screen.getByText('Track Custom'))

    expect(Analytics.trackCustom).toHaveBeenCalledWith('test', 'click', { page: 'home' })
  })

  it('tracks funnel progression events', async () => {
    const user = userEvent.setup()
    const { Analytics } = await import('@/lib/analytics')
    render(<AnalyticsProvider><TestComponent /></AnalyticsProvider>)

    await user.click(screen.getByText('Track Funnel'))

    expect(Analytics.trackFunnel.pricingView).toHaveBeenCalledWith('adaptive')
  })

  it('provides fallback no-op analytics when context is missing', () => {
    const HookProbe = () => {
      const analytics = useAnalyticsContext()
      analytics.trackConversion.tierReservation('core', 96)
      analytics.trackCustom('test', 'event')
      analytics.trackFunnel.pricingView()
      return null
    }
    expect(() => render(<HookProbe />)).not.toThrow()
  })
})
