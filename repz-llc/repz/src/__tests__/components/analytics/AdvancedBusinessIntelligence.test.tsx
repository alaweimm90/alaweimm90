import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

vi.mock('@/lib/analytics', () => ({
  Analytics: {
    trackCustom: vi.fn(() => Promise.resolve())
  }
}))

vi.mock('@/lib/orchestration/dag', () => ({
  buildAnalyticsDAG: vi.fn((params) => ({
    nodes: [
      { name: 'analytics-snapshots', traceId: params.traceId, timeframe: params.timeframe, enabled: true },
      { name: 'business-intelligence', traceId: params.traceId, timeframe: params.timeframe, enabled: true },
      { name: 'workflow-status', traceId: params.traceId, timeframe: params.timeframe, enabled: true },
      { name: 'cohort-analysis', traceId: params.traceId, timeframe: params.timeframe, enabled: true },
      { name: 'ab-testing-engine', traceId: params.traceId, timeframe: params.timeframe, enabled: true }
    ],
    edges: []
  }))
}))

let currentTestScenario = 'default';

vi.mock('@/lib/orchestration/executor', () => ({
  runAnalyticsPlan: vi.fn(async (plan, client) => {
    const hasTraceId = plan.nodes.some(n => n.traceId);
    
    if (currentTestScenario === 'error-message') {
      return {
        metrics: metricsPayload,
        isSnapshot: false,
        workflowStatus: 'running',
        workflowSteps: [{ step: 'bi_publish', status: 'failed', error_message: 'Snapshot insert error' }],
        cohortData: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } },
        abTestData: { activeTests: [], completedTests: [] }
      };
    }
    
    if (currentTestScenario === 'many-steps') {
      return {
        metrics: metricsPayload,
        isSnapshot: false,
        workflowStatus: 'running',
        workflowSteps: [
          { step: 's1', status: 'succeeded' },
          { step: 's2', status: 'succeeded' },
          { step: 's3', status: 'succeeded' },
          { step: 's4', status: 'succeeded' },
          { step: 's5', status: 'succeeded' },
          { step: 's6', status: 'succeeded' }
        ],
        cohortData: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } },
        abTestData: { activeTests: [], completedTests: [] }
      };
    }
    
    return {
      metrics: hasTraceId ? metricsPayload : metricsPayload, // Always return metrics for fallback
      isSnapshot: hasTraceId,
      snapshotCreatedAt: hasTraceId ? new Date().toISOString() : undefined,
      workflowStatus: 'running',
      workflowSteps: [{ step: 'checkout_completed', status: 'succeeded' }, { step: 'bi_publish', status: 'succeeded' }],
      cohortData: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } },
      abTestData: { activeTests: [], completedTests: [] }
    };
  })
}))

const metricsPayload = {
  revenue: {
    totalRevenue: 100000,
    revenueGrowth: { monthOverMonth: 5 },
    revenueByTier: { core: 1000, adaptive: 2000, performance: 3000, longevity: 4000 }
  },
  users: { activeUsers: 5000, newUsers: 200 },
  conversions: { trialToSubscription: 0.25 },
  churn: { overallChurnRate: 0.05 },
  growth: {
    customerAcquisitionCost: 100,
    lifetimeValue: 1000,
    marketingEfficiency: { ltv_cac_ratio: 10 },
    paybackPeriod: 6
  },
  insights: {
    keyFindings: ['Test finding'],
    recommendations: [{ action: 'Do X', impact: 'High', effort: 'Low', category: 'Test', timeline: 'now' }],
    alerts: [{ type: 'warning', message: 'Test alert' }]
  }
} as any

// Consolidated executor mock
vi.mock('@/lib/orchestration/executor', () => ({
  runAnalyticsPlan: vi.fn(async (plan) => {
    const hasTraceId = Array.isArray(plan?.nodes) && plan.nodes.some((n: { traceId?: string }) => !!n.traceId);
    if (currentTestScenario === 'error-message') {
      return {
        metrics: metricsPayload,
        isSnapshot: false,
        workflowStatus: 'running',
        workflowSteps: [{ step: 'snapshot_insert', status: 'failed', error_message: 'Snapshot insert error' }],
        cohortData: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } },
        abTestData: { activeTests: [], completedTests: [] }
      };
    }
    if (currentTestScenario === 'many-steps') {
      return {
        metrics: metricsPayload,
        isSnapshot: false,
        workflowStatus: 'running',
        workflowSteps: [
          { step: 's1', status: 'succeeded' },
          { step: 's2', status: 'succeeded' },
          { step: 's3', status: 'succeeded' },
          { step: 's4', status: 'succeeded' },
          { step: 's5', status: 'succeeded' },
          { step: 's6', status: 'succeeded' }
        ],
        cohortData: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } },
        abTestData: { activeTests: [], completedTests: [] }
      };
    }
    return {
      metrics: metricsPayload,
      isSnapshot: hasTraceId,
      snapshotCreatedAt: hasTraceId ? new Date().toISOString() : undefined,
      workflowStatus: 'running',
      workflowSteps: [
        { step: 'checkout_completed', status: 'succeeded' },
        { step: 'bi_publish', status: 'succeeded' }
      ],
      cohortData: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } },
      abTestData: { activeTests: [], completedTests: [] }
    };
  })
}))

describe('AdvancedBusinessIntelligence', () => {
  beforeEach(() => {
    vi.resetModules()
    vi.clearAllMocks()
  })

  it('loads snapshot metrics when trace_id is present', async () => {
    const { default: AdvancedBusinessIntelligence } = await import('@/components/analytics/AdvancedBusinessIntelligence')

    render(
      <MemoryRouter initialEntries={[`/dashboard?trace_id=test-trace&tab=analytics`]}> 
        <AdvancedBusinessIntelligence />
      </MemoryRouter>
    )

    expect(await screen.findByText('Advanced Business Intelligence')).toBeInTheDocument()
    expect(await screen.findByText('Total Revenue')).toBeInTheDocument()
    expect(await screen.findByText('Snapshot')).toBeInTheDocument()
  })

  it('falls back to live BI when no snapshot', async () => {
    const { default: AdvancedBusinessIntelligence } = await import('@/components/analytics/AdvancedBusinessIntelligence')

    render(
      <MemoryRouter initialEntries={[`/dashboard`]}> 
        <AdvancedBusinessIntelligence />
      </MemoryRouter>
    )

    expect(await screen.findByText('Advanced Business Intelligence')).toBeInTheDocument()
    expect(await screen.findByText('Total Revenue')).toBeInTheDocument()
    expect(screen.queryByText('Snapshot')).toBeNull()
  })

  it('refreshes snapshot on demand', async () => {
    const { default: AdvancedBusinessIntelligence } = await import('@/components/analytics/AdvancedBusinessIntelligence')
    const { supabase } = await import('@/integrations/supabase/client')
    supabase.functions.invoke = vi.fn((fn: string, opts?: any) => {
      if (fn === 'analytics-snapshots' && opts?.body?.action === 'refresh') {
        return Promise.resolve({ data: { data: { payload: metricsPayload } }, error: null })
      }
      if (fn === 'workflow-status') {
        return Promise.resolve({ data: { data: { workflow: { status: 'running' }, steps: [{ step: 'checkout_completed', status: 'succeeded' }, { step: 'bi_publish', status: 'succeeded' }] } }, error: null })
      }
      if (fn === 'cohort-analysis') {
        return Promise.resolve({ data: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } }, error: null })
      }
      if (fn === 'ab-testing-engine') {
        return Promise.resolve({ data: { activeTests: [], completedTests: [] }, error: null })
      }
      return Promise.resolve({ data: metricsPayload, error: null })
    }) as any

    render(
      <MemoryRouter initialEntries={[`/dashboard?trace_id=test-trace&tab=analytics`]}> 
        <AdvancedBusinessIntelligence />
      </MemoryRouter>
    )

    const btn = await screen.findByText('Refresh')
    fireEvent.click(btn)

    await waitFor(() => {
      expect(screen.getByText('Snapshot')).toBeInTheDocument()
      expect(screen.getByText('Workflow: running')).toBeInTheDocument()
      expect(screen.getByText('checkout completed')).toBeInTheDocument()
      expect(screen.getByText('bi publish')).toBeInTheDocument()
    })
  })

  it('updates snapshot history after refresh', async () => {
    const { default: AdvancedBusinessIntelligence } = await import('@/components/analytics/AdvancedBusinessIntelligence')
    const { supabase } = await import('@/integrations/supabase/client')
    const historyRows = [
      { id: 'h1', created_at: new Date().toISOString(), timeframe: '30d' },
      { id: 'h2', created_at: new Date().toISOString(), timeframe: '30d' },
    ]
    supabase.functions.invoke = vi.fn((fn: string, opts?: any) => {
      if (fn === 'workflow-status') {
        return Promise.resolve({ data: { data: { workflow: { status: 'running' }, steps: [] } }, error: null })
      }
      if (fn === 'analytics-snapshots' && opts?.body?.action === 'refresh') {
        return Promise.resolve({ data: { data: { payload: metricsPayload, created_at: new Date().toISOString() } }, error: null })
      }
      if (fn === 'analytics-snapshots' && opts?.body?.action === 'list') {
        return Promise.resolve({ data: { data: historyRows }, error: null })
      }
      if (fn === 'analytics-snapshots') {
        return Promise.resolve({ data: { data: { payload: metricsPayload, created_at: new Date().toISOString() } }, error: null })
      }
      if (fn === 'cohort-analysis') {
        return Promise.resolve({ data: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } }, error: null })
      }
      if (fn === 'ab-testing-engine') {
        return Promise.resolve({ data: { activeTests: [], completedTests: [] }, error: null })
      }
      return Promise.resolve({ data: metricsPayload, error: null })
    }) as any

    render(
      <MemoryRouter initialEntries={[`/dashboard?trace_id=t1&tab=analytics`]}> 
        <AdvancedBusinessIntelligence />
      </MemoryRouter>
    )

    const refresh = await screen.findByText('Refresh')
    fireEvent.click(refresh)

    await waitFor(() => {
      expect(screen.getByText('Workflow: running')).toBeInTheDocument()
      expect(screen.getByText('Snapshot 1')).toBeInTheDocument()
    })
  })

  it('copies Share link with timeframe', async () => {
    const { default: AdvancedBusinessIntelligence } = await import('@/components/analytics/AdvancedBusinessIntelligence')
    const { supabase } = await import('@/integrations/supabase/client')
    supabase.functions.invoke = vi.fn((fn: string) => {
      if (fn === 'analytics-snapshots') {
        return Promise.resolve({ data: { data: { payload: metricsPayload, created_at: new Date().toISOString() } }, error: null })
      }
      if (fn === 'cohort-analysis') {
        return Promise.resolve({ data: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } }, error: null })
      }
      if (fn === 'ab-testing-engine') {
        return Promise.resolve({ data: { activeTests: [], completedTests: [] }, error: null })
      }
      return Promise.resolve({ data: null, error: null })
    }) as any

    const writeText = vi.fn(() => Promise.resolve())
    Object.assign(navigator, { clipboard: { writeText } })

    render(
      <MemoryRouter initialEntries={[`/dashboard?trace_id=test-trace&tab=analytics&timeframe=90d`]}> 
        <AdvancedBusinessIntelligence />
      </MemoryRouter>
    )

    const btn = await screen.findByText('Share')
    fireEvent.click(btn)
    await waitFor(() => {
      expect(writeText).toHaveBeenCalled()
      const arg = writeText.mock.calls[0][0]
      expect(arg).toContain('trace_id=test-trace')
      expect(arg).toContain('timeframe=90d')
    })
  })

  it('loads snapshot history and selects a previous snapshot', async () => {
    const { default: AdvancedBusinessIntelligence } = await import('@/components/analytics/AdvancedBusinessIntelligence')
    const { supabase } = await import('@/integrations/supabase/client')
    const historyRows = [
      { id: 's1', created_at: new Date().toISOString(), timeframe: '30d' },
      { id: 's2', created_at: new Date().toISOString(), timeframe: '30d' },
    ]
    const payloadPrev = { ...metricsPayload, users: { activeUsers: 9999, newUsers: 111 } }
    supabase.functions.invoke = vi.fn((fn: string, opts?: any) => {
      if (fn === 'workflow-status') {
        return Promise.resolve({ data: { data: { workflow: { status: 'running' }, steps: [] } }, error: null })
      }
      if (fn === 'analytics-snapshots' && opts?.body?.action === 'list') {
        return Promise.resolve({ data: { data: historyRows }, error: null })
      }
      if (fn === 'analytics-snapshots') {
        return Promise.resolve({ data: { data: { payload: metricsPayload, created_at: new Date().toISOString() } }, error: null })
      }
      if (fn === 'cohort-analysis') {
        return Promise.resolve({ data: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } }, error: null })
      }
      if (fn === 'ab-testing-engine') {
        return Promise.resolve({ data: { activeTests: [], completedTests: [] }, error: null })
      }
      return Promise.resolve({ data: metricsPayload, error: null })
    }) as any

    const fromMock = vi.fn(() => ({
      select: vi.fn(() => ({ eq: vi.fn(() => ({ maybeSingle: vi.fn(() => Promise.resolve({ data: { payload: payloadPrev, created_at: new Date().toISOString() } }) ) })) }))
    }))
    // @ts-expect-error override
    supabase.from = fromMock

    render(
      <MemoryRouter initialEntries={[`/dashboard?trace_id=test-trace&tab=analytics`]}> 
        <AdvancedBusinessIntelligence />
      </MemoryRouter>
    )

    const historyBtn = await screen.findByText('History')
    fireEvent.click(historyBtn)

    const pickBtn = await screen.findByText('Snapshot 1')
    fireEvent.click(pickBtn)

    expect(await screen.findByText('Active Users')).toBeInTheDocument()
  })

  it('updates timeframe and reflects in shared link', async () => {
    const { default: AdvancedBusinessIntelligence } = await import('@/components/analytics/AdvancedBusinessIntelligence')
    const { supabase } = await import('@/integrations/supabase/client')
    supabase.functions.invoke = vi.fn((fn: string) => {
      if (fn === 'business-intelligence') {
        return Promise.resolve({ data: metricsPayload, error: null })
      }
      if (fn === 'cohort-analysis') {
        return Promise.resolve({ data: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } }, error: null })
      }
      if (fn === 'ab-testing-engine') {
        return Promise.resolve({ data: { activeTests: [], completedTests: [] }, error: null })
      }
      return Promise.resolve({ data: { data: null }, error: null })
    }) as any

    const writeText = vi.fn(() => Promise.resolve())
    Object.assign(navigator, { clipboard: { writeText } })

    render(
      <MemoryRouter initialEntries={[`/dashboard?tab=analytics`]}> 
        <AdvancedBusinessIntelligence />
      </MemoryRouter>
    )

    const btn = await screen.findByText('7d')
    fireEvent.click(btn)
    const share = await screen.findByText('Share')
    fireEvent.click(share)
    await waitFor(() => {
      expect(writeText).toHaveBeenCalled()
      const arg = writeText.mock.calls[0][0]
      expect(arg).toContain('timeframe=7d')
    })
  })

  it('exports metrics as JSON', async () => {
    const { default: AdvancedBusinessIntelligence } = await import('@/components/analytics/AdvancedBusinessIntelligence')
    const { supabase } = await import('@/integrations/supabase/client')
    supabase.functions.invoke = vi.fn((fn: string) => {
      if (fn === 'business-intelligence') {
        return Promise.resolve({ data: metricsPayload, error: null })
      }
      if (fn === 'cohort-analysis') {
        return Promise.resolve({ data: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } }, error: null })
      }
      if (fn === 'ab-testing-engine') {
        return Promise.resolve({ data: { activeTests: [], completedTests: [] }, error: null })
      }
      return Promise.resolve({ data: { data: null }, error: null })
    }) as any

    const click = vi.fn()
    const createObjectURL = vi.fn(() => 'blob:mock')
    URL.createObjectURL = createObjectURL as any
    HTMLAnchorElement.prototype.click = click as any

    render(
      <MemoryRouter initialEntries={[`/dashboard?timeframe=30d`]}> 
        <AdvancedBusinessIntelligence />
      </MemoryRouter>
    )

    const btn = await screen.findByText('Export')
    fireEvent.click(btn)

    expect(createObjectURL).toHaveBeenCalled()
    expect(click).toHaveBeenCalled()
  })

  it('renders workflow step error messages', async () => {
    currentTestScenario = 'error-message';
    const { default: AdvancedBusinessIntelligence } = await import('@/components/analytics/AdvancedBusinessIntelligence')

    render(
      <MemoryRouter initialEntries={[`/dashboard?trace_id=test-trace&tab=analytics`]}> 
        <AdvancedBusinessIntelligence />
      </MemoryRouter>
    )

    expect(await screen.findByText('Workflow: running')).toBeInTheDocument()
    expect(await screen.findByText('Snapshot insert error')).toBeInTheDocument()
    currentTestScenario = 'default';
  })

  it('shows all workflow steps when toggled', async () => {
    currentTestScenario = 'many-steps';
    const { default: AdvancedBusinessIntelligence } = await import('@/components/analytics/AdvancedBusinessIntelligence')

    render(
      <MemoryRouter initialEntries={[`/dashboard?trace_id=test-trace&tab=analytics`]}> 
        <AdvancedBusinessIntelligence />
      </MemoryRouter>
    )

    expect(await screen.findByText('Workflow: running')).toBeInTheDocument()
    expect(screen.queryByText('s6')).toBeNull()

    const btn = await screen.findByText('View Steps')
    fireEvent.click(btn)
    expect(await screen.findByText('s6')).toBeInTheDocument()
    currentTestScenario = 'default';
  })

  it('auto-refreshes snapshot when workflow succeeds', async () => {
    const { default: AdvancedBusinessIntelligence } = await import('@/components/analytics/AdvancedBusinessIntelligence')
    const { supabase } = await import('@/integrations/supabase/client')
    supabase.functions.invoke = vi.fn((fn: string, opts?: any) => {
      if (fn === 'workflow-status') {
        return Promise.resolve({ data: { data: { workflow: { status: 'succeeded' }, steps: [] } }, error: null })
      }
      if (fn === 'analytics-snapshots' && !opts?.body?.action) {
        return Promise.resolve({ data: { data: { payload: metricsPayload, created_at: new Date().toISOString() } }, error: null })
      }
      if (fn === 'analytics-snapshots' && opts?.body?.action === 'refresh') {
        return Promise.resolve({ data: { data: { payload: metricsPayload, created_at: new Date().toISOString() } }, error: null })
      }
      if (fn === 'cohort-analysis') {
        return Promise.resolve({ data: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } }, error: null })
      }
      if (fn === 'ab-testing-engine') {
        return Promise.resolve({ data: { activeTests: [], completedTests: [] }, error: null })
      }
      return Promise.resolve({ data: metricsPayload, error: null })
    }) as any

    render(
      <MemoryRouter initialEntries={[`/dashboard?trace_id=test-trace&tab=analytics`]}> 
        <AdvancedBusinessIntelligence />
      </MemoryRouter>
    )

    expect(await screen.findByText('Snapshot')).toBeInTheDocument()
  })

  it('disables Refresh when no trace_id', async () => {
    const { default: AdvancedBusinessIntelligence } = await import('@/components/analytics/AdvancedBusinessIntelligence')
    const { supabase } = await import('@/integrations/supabase/client')
    supabase.functions.invoke = vi.fn((fn: string) => {
      if (fn === 'business-intelligence') {
        return Promise.resolve({ data: metricsPayload, error: null })
      }
      if (fn === 'cohort-analysis') {
        return Promise.resolve({ data: { analysis: { retentionMetrics: { overallRetention: { day1: 0.9, day7: 0.7, day30: 0.5 } } }, insights: { keyFindings: [] } }, error: null })
      }
      if (fn === 'ab-testing-engine') {
        return Promise.resolve({ data: { activeTests: [], completedTests: [] }, error: null })
      }
      return Promise.resolve({ data: null, error: null })
    }) as any

    render(
      <MemoryRouter initialEntries={[`/dashboard`]}> 
        <AdvancedBusinessIntelligence />
      </MemoryRouter>
    )

    const btn = await screen.findByText('Refresh')
    expect(btn).toBeDisabled()
  })

})
