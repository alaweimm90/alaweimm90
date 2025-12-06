'use client'

import { useEffect } from 'react'
import { usePathname } from 'next/navigation'

declare global {
  interface Window {
    gtag: (
      command: 'config' | 'event',
      targetId: string,
      config?: Record<string, any>
    ) => void
  }
}

export function Analytics() {
  const pathname = usePathname()

  useEffect(() => {
    if (typeof window.gtag !== 'undefined') {
      window.gtag('config', process.env.NEXT_PUBLIC_GA_ID || '', {
        page_path: pathname,
      })
    }
  }, [pathname])

  useEffect(() => {
    // Track page views
    const handleRouteChange = (url: string) => {
      if (typeof window.gtag !== 'undefined') {
        window.gtag('event', 'page_view', {
          page_location: url,
          page_title: document.title,
        })
      }
    }

    // Initial page view
    handleRouteChange(window.location.href)

    // Listen for navigation events
    window.addEventListener('popstate', () => handleRouteChange(window.location.href))

    return () => {
      window.removeEventListener('popstate', () => handleRouteChange(window.location.href))
    }
  }, [])

  return null
}

export function trackEvent(
  action: string,
  category: string,
  label?: string,
  value?: number
) {
  if (typeof window.gtag !== 'undefined') {
    window.gtag('event', action, {
      event_category: category,
      event_label: label,
      value: value,
    })
  }
}