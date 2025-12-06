import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useTierAccess, useFeatureAccess } from '../useTierAccess'

// Mock AuthContext
const mockUser = {
  id: 'test-user-id',
  email: 'test@example.com'
}

vi.mock('@/contexts/AuthContext', () => ({
  useAuth: () => ({
    user: mockUser,
    isAuthenticated: true
  })
}))

// Mock tier constants
vi.mock('@/constants/tiers', () => ({
  TIER_CONFIGS: {
    core: {
      name: 'Core',
      features: ['progressPhotos', 'customWorkouts'],
      level: 1
    },
    adaptive: {
      name: 'Adaptive',
      features: ['progressPhotos', 'customWorkouts', 'coachAccess'],
      level: 2
    },
    performance: {
      name: 'Performance',
      features: ['progressPhotos', 'customWorkouts', 'coachAccess', 'formAnalysis'],
      level: 3
    },
    longevity: {
      name: 'Longevity',
      features: ['progressPhotos', 'customWorkouts', 'coachAccess', 'formAnalysis', 'live_coaching'],
      level: 4
    }
  }
}))

// Mock Supabase
const mockTierData = {
  subscription_tier: 'adaptive',
  tier_features: {
    progressPhotos: true,
    customWorkouts: true,
    coachAccess: true,
    formAnalysis: false,
    live_coaching: false
  }
}

vi.mock('@/integrations/supabase/client', () => ({
  supabase: {
    from: vi.fn(() => ({
      select: vi.fn(() => ({
        eq: vi.fn(() => ({
          single: vi.fn(() => Promise.resolve({
            data: mockTierData,
            error: null
          }))
        }))
      }))
    }))
  }
}))

describe('useTierAccess', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns correct tier access information', async () => {
    const { result } = renderHook(() => useTierAccess())

    await waitFor(() => {
      expect(result.current.userTier).toBe('adaptive')
      expect(result.current.hasMinimumTier).toBeDefined()
      expect(result.current.hasFeature).toBeDefined()
    })
  })

  it('provides correct feature access', async () => {
    const { result } = renderHook(() => useTierAccess())

    await waitFor(() => {
      expect(result.current.hasWeeklyCheckins).toBe(true)
      expect(result.current.responseTimeHours).toBe(48)
      expect(result.current.dashboardType).toBe('interactive')
    })
  })

  it('includes performance metrics', async () => {
    const { result } = renderHook(() => useTierAccess())

    await waitFor(() => {
      expect(result.current.responseTimeHours).toBeGreaterThan(0)
      expect(result.current.responseTimeHours).toBeLessThan(100)
    })
  })

  it('determines correct dashboard type', async () => {
    const { result } = renderHook(() => useTierAccess())

    await waitFor(() => {
      expect(result.current.dashboardType).toBe('interactive')
    })
  })
})

describe('useFeatureAccess', () => {
  it('correctly checks feature access', async () => {
    const { result } = renderHook(() => useTierAccess())

    await waitFor(() => {
      expect(result.current.hasFeature('weekly_checkins')).toBe(true)
    })
  })

  it('denies access to unavailable features', async () => {
    const { result } = renderHook(() => useTierAccess())

    await waitFor(() => {
      expect(result.current.hasFeature('peds')).toBe(false)
    })
  })
})
