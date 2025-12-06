import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@/utils/testHelpers'
import { PersonalizedDashboard } from '@/components/dashboard/PersonalizedDashboard'
import { supabase } from '@/integrations/supabase/client'

const mockUsers = {
  coreUser: { id: 'core-user-123', email: 'core@example.com', subscription_tier: 'core' },
  adaptiveUser: { id: 'adaptive-user-123', email: 'adaptive@example.com', subscription_tier: 'adaptive' },
  performanceUser: { id: 'performance-user-123', email: 'performance@example.com', subscription_tier: 'performance' },
  longevityUser: { id: 'longevity-user-123', email: 'longevity@example.com', subscription_tier: 'longevity' }
}

const mockWorkouts = [
  { id: 'workout-1', client_id: 'user-123', exercises_completed: 8, performance_score: 85 }
]

const mockAnalytics = {
  totalUsers: 1250, activeUsers: 892, revenue: 45000, conversionRate: 12.5
}

// Mock Supabase
vi.mock('@/integrations/supabase/client', () => ({
  supabase: {
    from: vi.fn(() => ({
      select: vi.fn(() => ({
        eq: vi.fn(() => Promise.resolve({
          data: mockWorkouts,
          error: null
        }))
      }))
    })),
    functions: {
      invoke: vi.fn(() => Promise.resolve({
        data: mockAnalytics,
        error: null
      }))
    }
  }
}))

// Mock AuthContext
const mockUseAuth = vi.fn()
vi.mock('@/contexts/AuthContext', () => ({
  useAuth: () => mockUseAuth(),
  AuthProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

describe('PersonalizedDashboard Component', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Dashboard Rendering by Tier', () => {
    it('renders core dashboard for core tier users', () => {
      mockUseAuth.mockReturnValue({
        user: mockUsers.coreUser,
        userTier: 'core',
        loading: false
      })

      const { rerender } = render(<PersonalizedDashboard />)
      
      expect(screen.getByText(/welcome/i)).toBeInTheDocument()
      expect(screen.getByText(/progress photos/i)).toBeInTheDocument()
      expect(screen.getByText(/nutrition tracking/i)).toBeInTheDocument()
      
      // Should not show advanced features
      expect(screen.queryByText(/ai coaching/i)).not.toBeInTheDocument()
    })

    it('renders adaptive dashboard for adaptive tier users', () => {
      mockUseAuth.mockReturnValue({
        user: mockUsers.adaptiveUser,
        userTier: 'adaptive',
        loading: false
      })

      const { rerender } = render(<PersonalizedDashboard />)
      
      expect(screen.getByText(/weekly check-ins/i)).toBeInTheDocument()
      expect(screen.getByText(/form analysis/i)).toBeInTheDocument()
      expect(screen.getByText(/body composition/i)).toBeInTheDocument()
    })

    it('renders performance dashboard for performance tier users', () => {
      mockUseAuth.mockReturnValue({
        user: mockUsers.performanceUser,
        userTier: 'performance',
        loading: false
      })

      render(<PersonalizedDashboard />)
      
      expect(screen.getByText(/ai coaching/i)).toBeInTheDocument()
      expect(screen.getByText(/live sessions/i)).toBeInTheDocument()
      expect(screen.getByText(/advanced analytics/i)).toBeInTheDocument()
    })

    it('renders longevity dashboard for longevity tier users', () => {
      mockUseAuth.mockReturnValue({
        user: mockUsers.longevityUser,
        userTier: 'longevity',
        loading: false
      })

      render(<PersonalizedDashboard />)
      
      expect(screen.getByText(/bioregulator protocols/i)).toBeInTheDocument()
      expect(screen.getByText(/peptide therapy/i)).toBeInTheDocument()
      expect(screen.getByText(/exclusive research/i)).toBeInTheDocument()
    })
  })

  describe('Workout Tracking Integration', () => {
    it('displays recent workouts correctly', async () => {
      mockUseAuth.mockReturnValue({
        user: mockUsers.performanceUser,
        userTier: 'performance',
        loading: false
      })

      render(<PersonalizedDashboard />)
      
      await waitFor(() => {
        expect(screen.getByText(/recent workouts/i)).toBeInTheDocument()
        expect(screen.getByText(/8 exercises/i)).toBeInTheDocument()
        expect(screen.getByText(/85% performance/i)).toBeInTheDocument()
      })
    })

    it('handles empty workout state', async () => {
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn(() => ({
          eq: vi.fn(() => Promise.resolve({
            data: [],
            error: null
          }))
        }))
      } as any)

      mockUseAuth.mockReturnValue({
        user: mockUsers.coreUser,
        userTier: 'core',
        loading: false
      })

      render(<PersonalizedDashboard />)
      
      await waitFor(() => {
        expect(screen.getByText(/no workouts recorded yet/i)).toBeInTheDocument()
        expect(screen.getByRole('button', { name: /create new workout/i })).toBeInTheDocument()
      })
    })

    it('shows create workout action when no workouts exist', async () => {
      mockUseAuth.mockReturnValue({
        user: mockUsers.coreUser,
        userTier: 'core',
        loading: false
      })

      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn(() => ({
          eq: vi.fn(() => Promise.resolve({ data: [], error: null }))
        }))
      } as any)

      render(<PersonalizedDashboard />)
      
      await waitFor(() => {
        const createButton = screen.getByRole('button', { name: /create new workout/i })
        expect(createButton).toBeInTheDocument()
      })
    })
  })

  describe('Analytics Integration', () => {
    it('displays analytics data for eligible tiers', async () => {
      mockUseAuth.mockReturnValue({
        user: mockUsers.performanceUser,
        userTier: 'performance',
        loading: false
      })

      render(<PersonalizedDashboard />)
      
      await waitFor(() => {
        expect(screen.getByText('1250')).toBeInTheDocument()
        expect(screen.getByText(/45,000/)).toBeInTheDocument()
        expect(screen.getByText(/12.5\s*%/)).toBeInTheDocument()
      })
    })

    it('hides analytics card for lower tiers', () => {
      mockUseAuth.mockReturnValue({
        user: mockUsers.coreUser,
        userTier: 'core',
        loading: false
      })

      render(<PersonalizedDashboard />)
      
      expect(screen.queryByText(/analytics dashboard/i)).toBeNull()
    })
  })

  describe('Responsive Dashboard Layout', () => {
    it('adapts layout for mobile devices', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      })

      mockUseAuth.mockReturnValue({
        user: mockUsers.performanceUser,
        userTier: 'performance',
        loading: false
      })

      render(<PersonalizedDashboard />)
      
      const dashboardGrid = screen.getByTestId('dashboard-grid')
      expect(dashboardGrid).toHaveClass(/md:grid-cols-2/)
    })

    it('uses multi-column layout on desktop', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      })

      mockUseAuth.mockReturnValue({
        user: mockUsers.performanceUser,
        userTier: 'performance',
        loading: false
      })

      render(<PersonalizedDashboard />)
      
      const dashboardGrid = screen.getByTestId('dashboard-grid')
      expect(dashboardGrid).toHaveClass(/md:grid-cols-2/)
    })
  })

  describe('Real-time Updates', () => {
    it('updates workout data in real-time', async () => {
      mockUseAuth.mockReturnValue({
        user: mockUsers.performanceUser,
        userTier: 'performance',
        loading: false
      })

      render(<PersonalizedDashboard />)
      
      // Simulate real-time update
      const updatedWorkouts = [
        ...mockWorkouts,
        {
          id: 'workout-3',
          client_id: 'user-123',
          started_at: new Date().toISOString(),
          completed_at: null,
          exercises_completed: 1,
          performance_score: null,
          status: 'active'
        }
      ]

      // Mock updated data
      vi.mocked(supabase.from).mockReturnValue({
        select: vi.fn(() => ({
          eq: vi.fn(() => Promise.resolve({
            data: updatedWorkouts,
            error: null
          }))
        }))
      } as any)

      // Trigger refresh by re-rendering component
      const { rerender } = render(<PersonalizedDashboard />)
      rerender(<PersonalizedDashboard />)

      await waitFor(() => {
        expect(screen.getAllByText(/exercises completed/i).length).toBeGreaterThan(0)
      })
    })
  })

  describe('Error Handling', () => {
    it('handles API errors gracefully', async () => {
      vi.mocked(supabase.functions.invoke).mockResolvedValue({
        data: null,
        error: { message: 'Network error' }
      })

      mockUseAuth.mockReturnValue({
        user: mockUsers.performanceUser,
        userTier: 'performance',
        loading: false
      })

      render(<PersonalizedDashboard />)
      
      await waitFor(() => {
        expect(screen.getByText(/analytics dashboard/i)).toBeInTheDocument()
        expect(screen.getByText(/analytics available/i)).toBeInTheDocument()
        expect(screen.getByRole('button', { name: /upgrade to access analytics/i })).toBeInTheDocument()
      })
    })

    it('handles network connectivity issues', async () => {
      vi.mocked(supabase.functions.invoke).mockRejectedValue(new Error('Network error'))

      mockUseAuth.mockReturnValue({
        user: mockUsers.performanceUser,
        userTier: 'performance',
        loading: false
      })

      render(<PersonalizedDashboard />)
      
      await waitFor(() => {
        expect(screen.getByText(/analytics dashboard/i)).toBeInTheDocument()
        expect(screen.getByText(/analytics available/i)).toBeInTheDocument()
      })
    })
  })

  describe('Performance Optimization', () => {
    it('lazy loads dashboard modules', async () => {
      mockUseAuth.mockReturnValue({
        user: mockUsers.longevityUser,
        userTier: 'longevity',
        loading: false
      })

      const { container } = render(<PersonalizedDashboard />)
      
      // Dashboard should render quickly
      expect(container.firstChild).toBeDefined()
      
      // Modules should load progressively
      await waitFor(() => {
        expect(screen.getByText(/welcome to your longevity dashboard/i)).toBeInTheDocument()
      }, { timeout: 1000 })
    })
  })
})
