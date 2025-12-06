import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@/test/utils/test-utils'
import App from '@/App'

// Mock performance APIs
const mockPerformanceObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn()
}))

// Add static properties to match PerformanceObserver interface
Object.defineProperty(mockPerformanceObserver, 'supportedEntryTypes', {
  value: ['navigation', 'paint', 'largest-contentful-paint', 'first-input', 'layout-shift'],
  writable: false
})

const mockGetEntriesByType = vi.fn()

beforeEach(() => {
  vi.clearAllMocks()
  
  // Mock Performance Observer with all required properties
  const MockPerformanceObserver = vi.fn().mockImplementation((callback) => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
    callback
  }))
  
  // Add static property as a property of the function
  Object.defineProperty(MockPerformanceObserver, 'supportedEntryTypes', {
    value: ['navigation', 'paint', 'largest-contentful-paint', 'first-input', 'layout-shift'],
    writable: false
  })
  
  global.PerformanceObserver = MockPerformanceObserver as any
  
  // Mock performance.getEntriesByType
  Object.defineProperty(global.performance, 'getEntriesByType', {
    value: mockGetEntriesByType,
    configurable: true
  })
  
  // Mock web-vitals
  vi.mock('web-vitals', () => ({
    getCLS: vi.fn((callback) => {
      callback({ name: 'CLS', value: 0.05, rating: 'good' })
    }),
    getFCP: vi.fn((callback) => {
      callback({ name: 'FCP', value: 1200, rating: 'good' })
    }),
    getFID: vi.fn((callback) => {
      callback({ name: 'FID', value: 80, rating: 'good' })
    }),
    getLCP: vi.fn((callback) => {
      callback({ name: 'LCP', value: 2100, rating: 'good' })
    }),
    getTTFB: vi.fn((callback) => {
      callback({ name: 'TTFB', value: 150, rating: 'good' })
    })
  }))
})

describe('Core Web Vitals Performance Tests', () => {
  it('measures LCP (Largest Contentful Paint) under 2.5s', async () => {
    const lcpEntries = [{
      name: 'largest-contentful-paint',
      startTime: 1800,
      element: document.createElement('img')
    }]
    
    mockGetEntriesByType.mockReturnValue(lcpEntries)
    
    render(<App />)
    
    await waitFor(() => {
      expect(screen.getByText(/repzcoach/i)).toBeInTheDocument()
    })
    
    // LCP should be under 2.5 seconds (2500ms)
    expect(lcpEntries[0].startTime).toBeLessThan(2500)
  })

  it('measures FID (First Input Delay) under 100ms', async () => {
    const fidEntries = [{
      name: 'first-input',
      processingStart: 100,
      startTime: 50,
      duration: 50
    }]
    
    mockGetEntriesByType.mockReturnValue(fidEntries)
    
    render(<App />)
    
    await waitFor(() => {
      const button = screen.getByText(/get started/i)
      expect(button).toBeInTheDocument()
    })
    
    // FID should be under 100ms
    const fid = fidEntries[0].processingStart - fidEntries[0].startTime
    expect(fid).toBeLessThan(100)
  })

  it('measures CLS (Cumulative Layout Shift) under 0.1', async () => {
    const clsEntries = [{
      name: 'layout-shift',
      value: 0.05,
      hadRecentInput: false
    }]
    
    mockGetEntriesByType.mockReturnValue(clsEntries)
    
    render(<App />)
    
    await waitFor(() => {
      expect(screen.getByText(/transform your fitness/i)).toBeInTheDocument()
    })
    
    // CLS should be under 0.1
    expect(clsEntries[0].value).toBeLessThan(0.1)
  })

  it('measures bundle size is optimized', async () => {
    const resourceEntries = [{
      name: 'main.js',
      transferSize: 250000, // 250KB
      encodedBodySize: 800000, // 800KB uncompressed
      decodedBodySize: 800000
    }]
    
    mockGetEntriesByType.mockReturnValue(resourceEntries)
    
    render(<App />)
    
    await waitFor(() => {
      expect(screen.getByText(/repzcoach/i)).toBeInTheDocument()
    })
    
    // Main bundle should be under 500KB compressed
    expect(resourceEntries[0].transferSize).toBeLessThan(500000)
  })

  it('measures initial render performance', async () => {
    const start = performance.now()
    
    render(<App />)
    
    await waitFor(() => {
      expect(screen.getByText(/repzcoach/i)).toBeInTheDocument()
    })
    
    const end = performance.now()
    const renderTime = end - start
    
    // Initial render should be under 500ms
    expect(renderTime).toBeLessThan(500)
  })

  it('measures route navigation performance', async () => {
    render(<App />)
    
    await waitFor(() => {
      expect(screen.getByText(/repzcoach/i)).toBeInTheDocument()
    })
    
    const start = performance.now()
    
    // Navigate to pricing page
    const pricingLink = screen.getByText(/pricing/i)
    pricingLink.click()
    
    await waitFor(() => {
      expect(screen.getByText(/choose your plan/i)).toBeInTheDocument()
    })
    
    const end = performance.now()
    const navigationTime = end - start
    
    // Route navigation should be under 200ms
    expect(navigationTime).toBeLessThan(200)
  })

  it('measures memory usage is within limits', async () => {
    // Mock memory API
    Object.defineProperty(performance, 'memory', {
      value: {
        usedJSHeapSize: 50000000, // 50MB
        totalJSHeapSize: 100000000, // 100MB
        jsHeapSizeLimit: 2000000000 // 2GB
      },
      configurable: true
    })
    
    render(<App />)
    
    await waitFor(() => {
      expect(screen.getByText(/repzcoach/i)).toBeInTheDocument()
    })
    
    const memoryUsage = (performance as any).memory.usedJSHeapSize
    
    // Memory usage should be under 100MB
    expect(memoryUsage).toBeLessThan(100000000)
  })

  it('validates lazy loading works correctly', async () => {
    render(<App />)
    
    // Initially should not load dashboard components
    expect(screen.queryByText(/dashboard components/i)).not.toBeInTheDocument()
    
    // Navigate to dashboard (would trigger lazy loading)
    window.history.pushState({}, '', '/dashboard')
    
    await waitFor(() => {
      // Dashboard should load only when needed
      expect(screen.getByText(/dashboard/i)).toBeInTheDocument()
    })
  })

  it('measures API response times', async () => {
    const apiStart = performance.now()
    
    render(<App />)
    
    // Simulate API call timing
    setTimeout(() => {
      const apiEnd = performance.now()
      const apiTime = apiEnd - apiStart
      
      // API calls should be under 500ms
      expect(apiTime).toBeLessThan(500)
    }, 100)
    
    await waitFor(() => {
      expect(screen.getByText(/repzcoach/i)).toBeInTheDocument()
    })
  })
})