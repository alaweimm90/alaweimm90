/**
 * Error Reporting Page
 * Shows the error reporting dashboard for monitoring application health
 */

import React, { useMemo } from 'react';
import { ErrorReportingDashboard } from '@/components/ErrorReportingDashboard';
import { PhysicsModuleHeader } from '@/components/PhysicsModuleHeader';
import { useSEO } from '@/hooks/use-seo';

export function ErrorReporting() {
  useSEO({ title: 'Error Reporting Dashboard – SimCore', description: 'Monitor application errors and performance metrics for debugging.' });
  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: 'Error Reporting Dashboard',
    description: 'Monitor application errors and performance metrics for debugging.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'error reporting, performance monitoring, debugging',
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);
  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <PhysicsModuleHeader
        title="Error Reporting Dashboard"
        description="Monitor application errors and performance metrics for better debugging and user experience"
        difficulty="Beginner"
        category="Development Tools"
      />
      
      <ErrorReportingDashboard />
    </div>
  );
}

export default ErrorReporting;